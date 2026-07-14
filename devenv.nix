{
  pkgs,
  ...
}:
{
  # The mstm/FaSTMM2 CLI binaries (built by the two derivations below, from
  # the exact same pinned commits pyMSTM's/pyFaSTMM's own flake.nix use)
  # land directly on PATH -- MstmCliAdapter/Fastmm2CliAdapter just shell
  # out to them by name. gfortran/lapack/blas are also needed here (in
  # addition to those two derivations) to build pymstm's and pyfastmm's
  # own compiled f2py extensions, since both are pulled in as local path
  # dependencies (see pyproject.toml's [tool.uv.sources]) and get built
  # by this project's own `uv sync`, not by their own devenv shells.
  packages = [
    (pkgs.callPackage ./nix/packages/mstm/package.nix { })
    (pkgs.callPackage ./nix/packages/fastmm2/package.nix { })
    pkgs.gfortran
    pkgs.lapack
    pkgs.blas
  ];

  env = {
    # UV_PYTHON = toString config.languages.python.package.interpreter;
    # LIBGL_ALWAYS_SOFTWARE = "1";
    # VTK_USE_X = "OFF";
    # VTK_DEFAULT_OPENGL_WINDOW = "vtkEGLRenderWindow";
    # PYVISTA_OFF_SCREEN = "true";
    # VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN = "1";
  };

  # scripts = {
  #   sl.exec = "uv run streamlit $@";
  # };

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
  '';

  languages.python = {
    enable = true;

    uv = {
      enable = true;
      sync = {
        enable = true;
        allExtras = true;
      };
    };

    libraries = with pkgs; [
      zlib
    ];
  };
}
