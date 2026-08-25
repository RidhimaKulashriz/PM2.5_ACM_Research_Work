# Spatial-surface and threshold result artifacts

The executed runner writes canonical CSV tables locally. The repository’s CSV paths are Git-LFS-managed, and GitHub browser upload cannot create LFS objects. Therefore, this directory publishes direct JSON review mirrors for each generated table. The JSON files are unchanged serializations of the local CSV outputs; no values or rows are transformed. The four PNG files under `plots/` are the static figures. Re-run the runner and validator from the repository root to regenerate and audit the complete local package.
