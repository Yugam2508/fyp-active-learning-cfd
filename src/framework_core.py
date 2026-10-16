"""
framework_core.py
=================
Verbatim extraction of the core classes from FYP_Active_Learning_Final.ipynb
(ToyProblem, Surrogate hierarchy, LCS, ActiveLearner) so the UQ extension can
be demonstrated standalone.  The ONLY change is that Latin Hypercube Sampling
uses scipy.stats.qmc instead of pyDOE (drop-in, same semantics).  All model
architectures, hyperparameters, and seeds are unchanged.
"""

import warnings
import numpy as np

try:                                    # torch preferred; sklearn MLP fallback
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:                        # pragma: no cover
    _HAS_TORCH = False
    print("[framework_core] torch unavailable -> DNNSurrogate uses "
          "sklearn MLPRegressor (same architecture: 2xReLU hidden, Adam, "
          "lr=1e-3, 800 epochs, standardised I/O)")

from scipy.stats import qmc, chi2
from scipy.spatial.distance import pdist, squareform
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(42)
if _HAS_TORCH:
    torch.manual_seed(42)


def lhs(d, n):
    """Latin Hypercube Sample, shape (n, d) in [0,1]^d  (scipy backend).

    The engine is seeded from the global numpy RNG so that np.random.seed()
    fixes the design.  Recent SciPy (>= 1.15) seeds an unseeded engine from
    OS entropy, which made initial designs irreproducible and broke the
    paired-seed protocol; see docs/REPRODUCIBILITY.md.
    """
    seed = int(np.random.randint(0, 2**31 - 1))
    try:
        engine = qmc.LatinHypercube(d=d, rng=np.random.default_rng(seed))
    except TypeError:                    # SciPy < 1.15 has no `rng` keyword
        engine = qmc.LatinHypercube(d=d, seed=seed)
    return engine.random(n)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark problems (unchanged from notebook)
# ─────────────────────────────────────────────────────────────────────────────
class ToyProblem:
    _BOUNDS = {
        "physics": None,
        "branin": [(-5, 10), (0, 15)],
        "hartmann3": (0.0, 1.0),
        "ishigami": (-np.pi, np.pi),
        "schwefel": (-500, 500),
        "friedman": (0.0, 1.0),
        "nonstationary": (-3.0, 3.0),
    }
    _FIXED_DIM = {"branin": 2, "hartmann3": 3, "ishigami": 3, "friedman": 5}

    def __init__(self, dim, name="physics"):
        self.dim = dim
        self.name = name.lower()
        b = self._BOUNDS[self.name]
        if self.name == "physics":
            self.bounds = [(-200, 200)] * dim
        elif isinstance(b, list):
            self.bounds = b
        else:
            self.bounds = [(b[0], b[1])] * dim

    def evaluate(self, X):
        return getattr(self, f"_{self.name}")(X)

    def noisy(self, X, sigma=0.1):
        return self.evaluate(X) + np.random.randn(len(X)) * sigma

    def _physics(self, X):
        x, y, z = X[:, 0], X[:, 1], X[:, 2]
        return np.sin(2*x)*np.cos(2*y) + np.exp(-z**2) + 0.2*(x**2 + y**2 + z**2)

    def _ishigami(self, X):
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        return np.sin(x1) + 7*np.sin(x2)**2 + 0.1*x3**4*np.sin(x1)

    def _schwefel(self, X):
        # Standard Schwefel function (Surjanovic & Bingham); added for the
        # domain-size control in experiments/campaign_domain.py.
        d = X.shape[1]
        return 418.9829*d - np.sum(X*np.sin(np.sqrt(np.abs(X))), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Surrogates (unchanged from notebook)
# ─────────────────────────────────────────────────────────────────────────────
class Surrogate:
    def __init__(self, name, family):
        self.name = name; self.family = family
    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError


class GPSurrogate(Surrogate):
    """GP member.  spec="orig" (default) is the notebook's kernel,
    k + White with NO signal-amplitude hyperparameter and an isotropic
    length scale.  spec="ard" adds a ConstantKernel amplitude and one length
    scale per input (ARD); its name carries a trailing "*".  The "orig" kernel
    is what causes the prior-reversion failure on the physics benchmark; see
    experiments/campaign_gpspec.py and docs/FINDINGS_soft_vs_hard.md."""
    def __init__(self, kernel="RBF", spec="orig"):
        super().__init__(f"GP-{kernel}" + ("*" if spec == "ard" else ""), "GP")
        self.kernel_name, self.spec = kernel, spec
        self.gp = None
        self.scaler = StandardScaler()
    def _kernel(self, d):
        if self.spec == "orig":
            k = RBF() if self.kernel_name == "RBF" else Matern(nu=1.5)
            return k + WhiteKernel(1e-2)
        ls = np.ones(d)
        k = (RBF(ls, (1e-2, 1e3)) if self.kernel_name == "RBF"
             else Matern(ls, (1e-2, 1e3), nu=1.5))
        return ConstantKernel(1.0, (1e-3, 1e3)) * k + WhiteKernel(1e-4, (1e-8, 1e-1))
    def fit(self, X, y):
        self.gp = GaussianProcessRegressor(
            kernel=self._kernel(X.shape[1]), normalize_y=True,
            n_restarts_optimizer=5)
        self.scaler.fit(X); self.gp.fit(self.scaler.transform(X), y)
    def predict(self, X):
        return self.gp.predict(self.scaler.transform(X))


class PCESurrogate(Surrogate):
    def __init__(self, degree):
        super().__init__(f"PCE-d{degree}", "PCE")
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree)),
            ("scaler", StandardScaler()),
            ("reg", BayesianRidge())])
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)


class RFSurrogate(Surrogate):
    def __init__(self, n=100):
        super().__init__(f"RF-{n}", "RF")
        self.model = RandomForestRegressor(n_estimators=n, random_state=42)
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)


class SVRSurrogate(Surrogate):
    def __init__(self, kernel="rbf", C=100):
        super().__init__(f"SVR-{kernel}", "SVR")
        self.model = Pipeline([
            ("scaler", StandardScaler()), ("svr", SVR(kernel=kernel, C=C))])
    def fit(self, X, y): self.model.fit(X, y)
    def predict(self, X): return self.model.predict(X)


class DNNSurrogate(Surrogate):
    def __init__(self, dim, hidden=64, tag=None):
        super().__init__(f"DNN-H{hidden}" + (f"-{tag}" if tag else ""), "DNN")
        self.dim = dim
        self.hidden = hidden
        if _HAS_TORCH:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)).float()
        else:
            from sklearn.neural_network import MLPRegressor
            self.net = MLPRegressor(hidden_layer_sizes=(hidden, hidden),
                                    activation="relu", solver="adam",
                                    learning_rate_init=1e-3, max_iter=800,
                                    tol=0.0, n_iter_no_change=800)

    def fit(self, X, y):
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
        self.ym, self.ys = y.mean(), y.std() + 1e-8
        if _HAS_TORCH:
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
            xm = torch.tensor(self.xm, dtype=torch.float32)
            xs = torch.tensor(self.xs, dtype=torch.float32)
            Xn = (Xt - xm)/xs; yn = (yt - self.ym)/self.ys
            opt = optim.Adam(self.net.parameters(), lr=1e-3)
            self.net.train()
            for _ in range(800):
                opt.zero_grad()
                loss = nn.MSELoss()(self.net(Xn), yn)
                loss.backward(); opt.step()
        else:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                self.net.fit((X - self.xm)/self.xs, (y - self.ym)/self.ys)

    def predict(self, X):
        if _HAS_TORCH:
            Xt = torch.tensor(X, dtype=torch.float32)
            xm = torch.tensor(self.xm, dtype=torch.float32)
            xs = torch.tensor(self.xs, dtype=torch.float32)
            self.net.eval()
            with torch.no_grad():
                y = self.net((Xt - xm)/xs).cpu().numpy().flatten()
            return y*self.ys + self.ym
        return self.net.predict((X - self.xm)/self.xs)*self.ys + self.ym


# ─────────────────────────────────────────────────────────────────────────────
# Hard LCS + ActiveLearner (unchanged from notebook)
# ─────────────────────────────────────────────────────────────────────────────
class LCS:
    def __init__(self, alpha=0.99, min_points=8):
        self.alpha = alpha
        self.min_points = min_points

    def score_models(self, models, X, y):
        preds = np.array([m.predict(X) for m in models])
        ensemble_mean = preds.mean(axis=0)
        sigma2 = np.var(y - ensemble_mean) + 1e-8
        dof = max(len(y) - 1, 1)
        scores = {m: np.sum((y - preds[i])**2 / sigma2)/dof
                  for i, m in enumerate(models)}
        return scores, dof

    def reject_one(self, models, X, y):
        if len(y) < self.min_points:
            return None, {"decision": "insufficient-data", "scores": {}, "threshold": None}
        scores, dof = self.score_models(models, X, y)
        threshold = chi2.ppf(self.alpha, dof) / dof
        worst = max(scores, key=scores.get)
        if scores[worst] > threshold:
            return worst, {"decision": "reject-one", "scores": scores,
                           "threshold": threshold, "rejected": worst}
        return None, {"decision": "keep-all", "scores": scores, "threshold": threshold}


class ActiveLearner:
    def __init__(self, problem, domain, n_init=5):
        self.problem = problem
        self.domain = np.array(domain)
        self.dim = self.domain.shape[0]
        X_lhs = lhs(self.dim, n_init)
        self.X = self.domain[:, 0] + X_lhs*(self.domain[:, 1] - self.domain[:, 0])
        self.y = problem.noisy(self.X)
        self.models = self._build_models()
        self.lcs = LCS(alpha=0.99)
        self.rejected = set()

    def _build_models(self):
        return [
            GPSurrogate("RBF"), GPSurrogate("Matern"),
            PCESurrogate(2), PCESurrogate(3),
            RFSurrogate(50), RFSurrogate(100),
            SVRSurrogate("rbf"),
            DNNSurrogate(self.dim, 64),
            DNNSurrogate(self.dim, 128),
        ]

    def train(self):
        for m in self.models:
            if m not in self.rejected:
                m.fit(self.X, self.y)

    def ueigv(self, grid):
        preds = np.array([m.predict(grid)
                          for m in self.models if m not in self.rejected])
        if preds.shape[0] < 2:
            return np.ones(len(grid))
        K, N = preds.shape
        scores = np.zeros(N)
        preds_norm = ((preds - preds.mean(axis=0, keepdims=True))
                      / (preds.std(axis=0, keepdims=True) + 1e-8))
        for i in range(N):
            p = preds_norm[:, i:i+1]
            d2 = squareform(pdist(p, metric="sqeuclidean"))
            A = np.exp(-d2/2.0)
            D = np.diag(A.sum(axis=1))
            Di = np.diag(1.0/np.sqrt(np.diag(D) + 1e-12))
            L = np.eye(K) - Di @ A @ Di
            scores[i] = np.sum(np.maximum(0.0, 1.0 - np.linalg.eigvalsh(L)))
        return scores

    def step(self, verbose=False):
        grid = np.random.uniform(self.domain[:, 0], self.domain[:, 1],
                                 size=(300, self.dim))
        ue = self.ueigv(grid)
        idx = np.argmax(ue)
        x_new = grid[idx:idx+1]
        y_new = self.problem.noisy(x_new)
        self.X = np.vstack([self.X, x_new])
        self.y = np.append(self.y, y_new)
        active = [m for m in self.models if m not in self.rejected]
        if len(active) > 1:
            rej, info = self.lcs.reject_one(active, self.X, self.y)
            if rej is not None:
                self.rejected.add(rej)
        return x_new, y_new
