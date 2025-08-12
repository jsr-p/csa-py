install:
    uv sync --all-groups
    uv pip install -e /home/jsr-p/gh-repos/documentation-stuff/sphinx-autodoc2

data_test:
    # simulate unbalanced
    python scripts/sim_test.py simub

    # simulate misc test data from R
    Rscript scripts/data/export_data.R
    Rscript scripts/data/panel_test.R
    python scripts/comparison/fastdid_res.py
    Rscript scripts/comparison/fastdid_res.R

    Rscript scripts/data/drdid_rc.R
    Rscript scripts/data/oreg_mpdta.R
    Rscript scripts/data/oreg_xvars.R
    # requires data/testing/sim_ub.csv
    Rscript scripts/data/oreg_ub.R

bench-csa:
    python scripts/sim_test.py datasets
    bash scripts/benchmark/bench_did.sh
    Rscript scripts/benchmark/plot_fastdid.R

bench-fastdid:
    python scripts/benchmark/bmarkfastdid.py
    Rscript scripts/benchmark/fastdidbenchcomp.R
    Rscript scripts/benchmark/plot_fastdid.R

prof-csa:
    bash scripts/profile/prof.sh 5

sphinx: 
    cp README.md docs/source
    # mkdir -p docs/source/figs
    # copy example images to docs
    # cp figs/*.png docs/source/figs/
    # cp figs/*.svg docs/source/figs/
    just sphinx-clean 
    just sphinx-build
    
sphinx-build:
    sphinx-build docs/source docs/build

sphinx-clean:
    sphinx-build -M clean docs/source docs/build
    rm docs/source/did_imp.rst || true
    rm docs/source/modules.rst || true
    rm -rf docs/source/apidocs/ || true

demo:
    # run this yourself and then run `ffmpeg` command
    # wf-recorder -g "0,24 960x300" -f demo/demo-pbar.mp4
    ffmpeg -i demo/demo-pbar.mp4 -vcodec libx264 -crf 30 -preset veryslow -an demo/demo-pbar-small.mp4
