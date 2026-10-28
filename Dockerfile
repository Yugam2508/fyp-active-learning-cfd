# Dockerfile — the software environment of the campaigns, for long-term
# reproducibility (docs/REPRODUCIBILITY.md, section 9): Ubuntu 24.04, OpenFOAM
# v1912 from the Ubuntu archive, Python 3.11 and the locked package versions,
# LaTeX for the papers. The result records are unpacked from results/*.zip.
#   docker build -t fyp-surrogate-gum .
#   docker run --rm fyp-surrogate-gum                      # the CI checks
#   docker run --rm -it -v "$PWD/results:/work/results" fyp-surrogate-gum bash
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CFD_WORKDIR=/tmp
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openfoam=1912.200626-2build3 libopenfoam=1912.200626-2build3 \
      ca-certificates git \
      texlive-latex-extra texlive-publishers texlive-fonts-extra \
      texlive-science texlive-pictures \
 && rm -rf /var/lib/apt/lists/*

# Python 3.11 from the deadsnakes PPA (the study used 3.11.15-1+noble1) and
# the locked packages with the CPU build of PyTorch, installed as in the CI
# workflow.
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common gpg-agent \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get install -y --no-install-recommends python3.11 python3.11-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /work
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt \
      --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .
RUN python experiments/results_pack.py
CMD ["sh", "-c", "python -m pytest -q && python experiments/check_benchmarks.py \
 && python experiments/test_reproducibility.py --quick --rtol 1e-4 \
 && python paper/imeko2027/check_numbers.py \
 && python experiments/analyse_journal.py --check \
 && python experiments/analyse_completion.py --check \
 && python paper/check_numbers_journal.py"]
