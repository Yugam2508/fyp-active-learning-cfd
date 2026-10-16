"""
cfd_dispersion.py — parametrised 2-D RANS dispersion case (OpenFOAM v1912).

Steady, neutral atmospheric surface layer flowing over an isolated building of
height H (streamwise width 10 m, two-dimensional, i.e. infinitely wide), with
a continuous ground-level release 20 m upstream of the windward face. The
quantity of interest is the concentration at breathing height (1.5 m) at a
receptor a distance x_r downstream of the leeward face, i.e. in or beyond the
recirculating wake. Compared with the Gaussian plume formula it adds the
features of a CFD solution the plume lacks: separation and recirculation
behind the building, a wake that changes with H, turbulent mixing set by the
turbulence model and the turbulent Schmidt number, and discretisation and
convergence error that varies with the inputs.

Inputs (uniform)                       range
  U_ref  wind speed at 10 m (m/s)      1 - 6
  z0     aerodynamic roughness (m)     0.01 - 0.3
  H      building height (m)           6 - 15
  Sc_t   turbulent Schmidt number      0.5 - 1.3
  x_r    receptor distance (m)         5 - 120  (from the leeward face)
  Q      release rate (g s^-1 m^-1)    1 - 20   (per metre crosswind)
Output: concentration (mg m^-3), positive.

Model. Flow and scalar are solved together by buoyantBoussinesqSimpleFoam
with zero gravity and zero expansion coefficient, so that its temperature
equation is the steady transport equation of a passive scalar s,
  div(phi s) - div((nu/Sc + nu_t/Sc_t) grad s) = source,
with Pr = Sc = 0.7 and Pr_t = Sc_t (the solvers' function objects, the usual
route to a passive scalar, fail in the Debian/Ubuntu build of v1912, and
scalarTransportFoam accepts only a constant diffusivity). Turbulence: standard
k-epsilon with sigma_eps = 1.11, consistent with the log-law inflow
(Richards & Hoxey 1993): U = u*/kappa ln((z+z0)/z0), k = u*^2/sqrt(Cmu),
eps = u*^3/(kappa (z+z0)) (atmBoundaryLayerInlet* conditions); rough ground
wall function (nutkAtmRoughWallFunction), smooth building walls, slip top.
Release: unit source (1 g s^-1 in the 1 m deep 2-D domain, i.e. per metre
crosswind) in the cell containing (-20, 0.5) m. As the scalar is passive,
C = 1000 Q s exactly, so Q needs no separate run. Domain: x from -80 m to
250 m (windward face at x = 0), height 150 m, 7776 cells.

run_case(U_ref, z0, H, Sc_t, workdir, refine=1) runs blockMesh and the solver
(refine multiplies the cell counts in both directions, for grid checks) and
returns s(x) at z = 1.5 m downstream of the building (cell-centre columns,
linear interpolation in z); receptor_concentration(profile, x_r, Q)
interpolates it in x.
"""

import os, re, shutil, subprocess
import numpy as np

X_IN, X_WW, W_B, X_OUT, Z_TOP = -80.0, 0.0, 10.0, 250.0, 150.0
X_SRC, Z_SRC = -20.0, 0.5                      # release location (m)
Z_REC = 1.5                                    # receptor height (m)
NX = (40, 16, 110)                             # cells: upstream, building, wake
NZ = (12, 36)                                  # cells: below roof, above roof
GX = (0.1, 1.0, 12.0)                          # last/first cell size per block
GZ = (1.0, 15.0)
NU, SC_LAM = 1.5e-5, 0.7
KAPPA, CMU = 0.41, 0.09

HEADER = """FoamFile
{{
    version 2.0;
    format  ascii;
    class   {cls};
    object  {obj};
}}
"""


def _w(path, cls, obj, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(HEADER.format(cls=cls, obj=obj) + body)


def _block_mesh(H, refine=1):
    xs = (X_IN, X_WW, X_WW + W_B, X_OUT)
    zs = (0.0, H, Z_TOP)
    v = lambda i, k, j: (k * 4 + i) + 12 * j
    verts = [(xs[i], float(j), zs[k]) for j in (0, 1) for k in range(3) for i in range(4)]
    blocks = []
    for k in range(2):
        for i in range(3):
            if (i, k) == (1, 0):
                continue                                          # the building
            hexv = (v(i, k, 0), v(i + 1, k, 0), v(i + 1, k, 1), v(i, k, 1),
                    v(i, k + 1, 0), v(i + 1, k + 1, 0), v(i + 1, k + 1, 1), v(i, k + 1, 1))
            blocks.append(f"    hex ({' '.join(map(str, hexv))}) ({NX[i] * refine} 1 {NZ[k] * refine}) "
                          f"simpleGrading ({GX[i]} 1 {GZ[k]})")
    P = np.array(verts)

    def face(a, b, c, d, outward):
        q = [a, b, c, d]
        n = np.cross(P[b] - P[a], P[c] - P[a])
        if np.dot(n, outward) < 0:
            q = q[::-1]
        return f"            ({' '.join(map(str, q))})"

    xm, zm = np.array([-1, 0, 0]), np.array([0, 0, -1])
    patches = {
        "inlet": ("patch", [face(v(0, k, 0), v(0, k, 1), v(0, k + 1, 1), v(0, k + 1, 0), xm)
                            for k in range(2)]),
        "outlet": ("patch", [face(v(3, k, 0), v(3, k, 1), v(3, k + 1, 1), v(3, k + 1, 0), -xm)
                             for k in range(2)]),
        "top": ("patch", [face(v(i, 2, 0), v(i + 1, 2, 0), v(i + 1, 2, 1), v(i, 2, 1), -zm)
                          for i in range(3)]),
        "ground": ("wall", [face(v(i, 0, 0), v(i + 1, 0, 0), v(i + 1, 0, 1), v(i, 0, 1), zm)
                            for i in (0, 2)]),
        "building": ("wall", [
            face(v(1, 0, 0), v(1, 0, 1), v(1, 1, 1), v(1, 1, 0), -xm),   # windward
            face(v(1, 1, 0), v(2, 1, 0), v(2, 1, 1), v(1, 1, 1), zm),    # roof
            face(v(2, 0, 0), v(2, 0, 1), v(2, 1, 1), v(2, 1, 0), xm)]),  # leeward
    }
    body = "scale 1;\n\nvertices\n(\n"
    body += "\n".join(f"    ({x:.6g} {y:.6g} {z:.6g})" for x, y, z in verts)
    body += "\n);\n\nblocks\n(\n" + "\n".join(blocks) + "\n);\n\nedges ();\n\nboundary\n(\n"
    for name, (typ, faces) in patches.items():
        body += f"    {name}\n    {{\n        type {typ};\n        faces\n        (\n"
        body += "\n".join(faces) + "\n        );\n    }\n"
    body += ("    frontAndBack\n    {\n        type empty;\n        faces\n        (\n"
             + "\n".join(f"            ({v(i, k, 0)} {v(i, k + 1, 0)} {v(i + 1, k + 1, 0)} {v(i + 1, k, 0)})"
                         for k in range(2) for i in range(3) if (i, k) != (1, 0))
             + "\n" + "\n".join(f"            ({v(i, k, 1)} {v(i + 1, k, 1)} {v(i + 1, k + 1, 1)} {v(i, k + 1, 1)})"
                                for k in range(2) for i in range(3) if (i, k) != (1, 0))
             + "\n        );\n    }\n);\n\nmergePatchPairs ();\n")
    return body


def _bc(inlet, outlet, top, ground, building):
    pat = lambda name, body: f"    {name}\n    {{\n{body}    }}\n"
    return ("boundaryField\n{\n" + pat("inlet", inlet) + pat("outlet", outlet)
            + pat("top", top) + pat("ground", ground) + pat("building", building)
            + pat("frontAndBack", "        type empty;\n") + "}\n")


def _fields(U_ref, z0):
    ustar = KAPPA * U_ref / np.log((10.0 + z0) / z0)
    k0 = ustar ** 2 / np.sqrt(CMU)
    eps10 = ustar ** 3 / (KAPPA * (10.0 + z0))
    abl = (f"        flowDir (1 0 0);\n        zDir (0 0 1);\n        Uref {U_ref:.8g};\n"
           f"        Zref 10;\n        z0 uniform {z0:.8g};\n        zGround uniform 0;\n"
           f"        kappa {KAPPA};\n        Cmu {CMU};\n")
    t = lambda typ, extra="": f"        type {typ};\n{extra}"
    val = lambda v: f"        value uniform {v};\n"
    zg = t("zeroGradient")
    F = {}
    F["U"] = ("volVectorField", f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform ({U_ref:.8g} 0 0);\n"
              + _bc(t("atmBoundaryLayerInletVelocity", abl + val(f"({U_ref:.8g} 0 0)")),
                    t("inletOutlet", "        inletValue uniform (0 0 0);\n" + val("(0 0 0)")),
                    t("slip"), t("noSlip"), t("noSlip")))
    F["p_rgh"] = ("volScalarField", "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n"
                  + _bc(zg, t("fixedValue", val(0)), zg,
                        t("fixedFluxPressure", val(0)), t("fixedFluxPressure", val(0))))
    calc0 = t("calculated", val(0))
    F["p"] = ("volScalarField", "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n"
              + _bc(calc0, calc0, calc0, calc0, calc0))
    F["k"] = ("volScalarField", f"dimensions [0 2 -2 0 0 0 0];\ninternalField uniform {k0:.8g};\n"
              + _bc(t("atmBoundaryLayerInletK", abl + val(f"{k0:.8g}")), zg, zg,
                    t("kqRWallFunction", val(f"{k0:.8g}")), t("kqRWallFunction", val(f"{k0:.8g}"))))
    F["epsilon"] = ("volScalarField", f"dimensions [0 2 -3 0 0 0 0];\ninternalField uniform {eps10:.8g};\n"
                    + _bc(t("atmBoundaryLayerInletEpsilon", abl + val(f"{eps10:.8g}")), zg, zg,
                          t("epsilonWallFunction", val(f"{eps10:.8g}")),
                          t("epsilonWallFunction", val(f"{eps10:.8g}"))))
    F["nut"] = ("volScalarField", "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\n"
                + _bc(calc0, calc0, calc0,
                      t("nutkAtmRoughWallFunction", f"        z0 uniform {z0:.8g};\n" + val(0)),
                      t("nutkWallFunction", val(0))))
    F["alphat"] = ("volScalarField", "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\n"
                   + _bc(calc0, calc0, calc0, calc0, calc0))
    F["T"] = ("volScalarField", "dimensions [0 0 0 1 0 0 0];\ninternalField uniform 0;\n"
              + _bc(t("fixedValue", val(0)),
                    t("inletOutlet", "        inletValue uniform 0;\n" + val(0)), zg, zg, zg))
    return F


CONTROL = """application buoyantBoussinesqSimpleFoam;
libs ("libatmosphericModels.so");
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {max_iter};
deltaT 1;
writeControl timeStep;
writeInterval {max_iter};
purgeWrite 0;
writeFormat ascii;
writePrecision 10;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable false;
"""

SCHEMES = """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,T) bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
"""

SOLUTION = """solvers
{
    p_rgh
    {
        solver GAMG;
        smoother GaussSeidel;
        tolerance 1e-8;
        relTol 0.05;
    }
    "(U|k|epsilon|T)"
    {
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-10;
        relTol 0.05;
    }
}
SIMPLE
{
    momentumPredictor yes;
    nNonOrthogonalCorrectors 0;
    pRefCell 0;
    pRefValue 0;
    residualControl
    {
        p_rgh 1e-5;
        U 1e-6;
        "(k|epsilon|T)" 1e-6;
    }
}
relaxationFactors
{
    fields
    {
        p_rgh 0.3;
    }
    equations
    {
        U 0.7;
        ".*" 0.7;
    }
}
"""

FVOPTIONS = f"""release
{{
    type            scalarSemiImplicitSource;
    active          yes;
    scalarSemiImplicitSourceCoeffs
    {{
        selectionMode   points;
        points          (({X_SRC} 0.5 {Z_SRC}));
        volumeMode      absolute;
        injectionRateSuSp
        {{
            T           (1 0);
        }}
    }}
}}
"""


def write_case(d, U_ref, z0, H, Sc_t, max_iter=4000, refine=1):
    if os.path.exists(d):
        shutil.rmtree(d)
    _w(f"{d}/system/blockMeshDict", "dictionary", "blockMeshDict", _block_mesh(H, refine))
    _w(f"{d}/system/controlDict", "dictionary", "controlDict", CONTROL.format(max_iter=max_iter))
    _w(f"{d}/system/fvSchemes", "dictionary", "fvSchemes", SCHEMES)
    _w(f"{d}/system/fvSolution", "dictionary", "fvSolution", SOLUTION)
    _w(f"{d}/constant/fvOptions", "dictionary", "fvOptions", FVOPTIONS)
    _w(f"{d}/constant/transportProperties", "dictionary", "transportProperties",
       f"transportModel Newtonian;\nnu {NU};\nbeta 0;\nTRef 0;\nPr {SC_LAM};\nPrt {Sc_t:.8g};\n")
    _w(f"{d}/constant/g", "uniformDimensionedVectorField", "g",
       "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 0);\n")
    _w(f"{d}/constant/radiationProperties", "dictionary", "radiationProperties",
       "radiation off;\nradiationModel none;\n")
    _w(f"{d}/constant/turbulenceProperties", "dictionary", "turbulenceProperties",
       "simulationType RAS;\nRAS\n{\n    RASModel kEpsilon;\n    turbulence on;\n"
       "    printCoeffs off;\n    kEpsilonCoeffs\n    {\n        Cmu 0.09;\n        C1 1.44;\n"
       "        C2 1.92;\n        C3 0;\n        sigmak 1.0;\n        sigmaEps 1.11;\n    }\n}\n")
    for name, (cls, body) in _fields(U_ref, z0).items():
        _w(f"{d}/0/{name}", cls, name, body)


# ── reading OpenFOAM ascii output ─────────────────────────────────────────
def _list_body(text):
    """Contents of the first top-level list '<n>\\n(' ... ')' after the header."""
    m = re.search(r"\n(\d+)\s*\n\(", text)
    n = int(m.group(1))
    return n, text[m.end():]


def _read_points(d):
    n, body = _list_body(open(f"{d}/constant/polyMesh/points").read())
    vals = np.array(re.findall(r"\(([^()]+)\)", body)[:n], dtype=object)
    return np.array([list(map(float, v.split())) for v in vals])


def _read_faces(d):
    n, body = _list_body(open(f"{d}/constant/polyMesh/faces").read())
    return [list(map(int, f.split())) for f in re.findall(r"\d+\(([^()]+)\)", body)[:n]]


def _read_labels(path):
    n, body = _list_body(open(path).read())
    return np.array(body.split(")")[0].split()[:n], dtype=int)


def cell_centres(d):
    """Cell centres as the mean of the cell's face centres (exact for boxes)."""
    P = _read_points(d)
    faces = _read_faces(d)
    own = _read_labels(f"{d}/constant/polyMesh/owner")
    nei = _read_labels(f"{d}/constant/polyMesh/neighbour")
    fc = np.array([P[f].mean(axis=0) for f in faces])
    n_cells = own.max() + 1
    s = np.zeros((n_cells, 3)); c = np.zeros(n_cells)
    np.add.at(s, own, fc); np.add.at(c, own, 1)
    np.add.at(s, nei, fc[:len(nei)]); np.add.at(c, nei, 1)
    return s / c[:, None]


def read_internal_scalar(path):
    text = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(", text)
    n = int(m.group(1))
    return np.array(text[m.end():].split(")")[0].split()[:n], dtype=float)


def _levels(v, tol=1e-6):
    """Distinct coordinate levels of a tensor grid (to tol) and each value's index."""
    u = np.sort(v)
    start = np.r_[True, np.diff(u) > tol]
    grp = np.cumsum(start) - 1
    levels = np.bincount(grp, u) / np.bincount(grp)
    idx = np.searchsorted((levels[1:] + levels[:-1]) / 2, v)
    return levels, idx


def run_case(U_ref, z0, H, Sc_t, workdir, keep=False, max_iter=4000, refine=1):
    """Run one case; returns the receptor-height profile and run diagnostics."""
    write_case(workdir, U_ref, z0, H, Sc_t, max_iter, refine)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    env.setdefault("WM_PROJECT_DIR", "/usr/share/openfoam")   # Debian/Ubuntu package
    for cmd in ("blockMesh", "buoyantBoussinesqSimpleFoam"):
        with open(os.path.join(workdir, f"log.{cmd}"), "w") as log:
            subprocess.run([cmd, "-case", workdir], stdout=log, stderr=subprocess.STDOUT,
                           env=env, check=True)
    times = [t for t in os.listdir(workdir) if re.fullmatch(r"[0-9.e+-]+", t) and t != "0"]
    t_end = max(times, key=float)
    log = open(os.path.join(workdir, "log.buoyantBoussinesqSimpleFoam")).read()
    C = cell_centres(workdir)
    s = read_internal_scalar(os.path.join(workdir, t_end, "T"))
    # wake columns below roof height: a tensor grid of cell centres
    sel = (C[:, 0] > X_WW + W_B) & (C[:, 2] < H)
    xs, ix = _levels(C[sel, 0]); zs, iz = _levels(C[sel, 2])
    grid = np.full((len(xs), len(zs)), np.nan)
    grid[ix, iz] = s[sel]
    assert sel.sum() == grid.size and not np.isnan(grid).any(), \
        "wake cells do not form a tensor grid"
    prof = np.array([np.interp(Z_REC, zs, col) for col in grid])
    out = {"x": (xs - (X_WW + W_B)).tolist(), "s": prof.tolist(),
           "iterations": int(float(t_end)),
           "converged": "solution converged in" in log,
           "s_min": float(s.min())}
    if not keep:
        shutil.rmtree(workdir)
    return out


def receptor_concentration(profile, x_r, Q):
    """C (mg m^-3) at x_r metres downstream of the leeward face, release Q g/s/m.

    The case is run with a unit source (1 g/s per metre of the 1 m deep 2-D
    domain), so s is in g m^-3 per (g s^-1 m^-1) and C = 1000 Q s."""
    return 1e3 * Q * np.interp(x_r, profile["x"], profile["s"])
