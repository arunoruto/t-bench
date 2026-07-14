{
  description = "Benchmark harness comparing MSTM and FaSTMM2 (Python wrappers and CLI binaries) on the same cluster configurations";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0";

    # pymstm's/pyfastmm's own flake.nix only build the standalone CLI
    # reference binaries so far, not the Python packages themselves (see
    # the comment on `mstm`/`fastmm2` in those files) -- for now, build
    # the Python packages straight from the sibling checkouts via a local
    # path input. Once pyMSTM's/pyFaSTMM's own flake.nix grow a
    # `packages.pymstm`/`packages.pyfastmm` output, the `pymstm`/
    # `pyfastmm` derivations below should be deleted and replaced with
    # `inputs.pymstm.url = "path:/home/mar/Development/pyMSTM"; ...` +
    # `packages.pymstm = pymstm.packages.${system}.pymstm;` -- t-bench
    # would only ever *reference* those builds, never redefine them here.
    pymstm-repo = {
      url = "path:/home/mar/Development/pyMSTM";
      flake = false;
    };
    # Same pinned commit as pyMSTM's own flake.nix's `mstm-src` input --
    # keep these in sync if that one ever moves.
    mstm-src = {
      url = "github:dmckwski/MSTM/a0c982121cf9ac352531f4816639a07d814385bd";
      flake = false;
    };

    pyfastmm-repo = {
      url = "path:/home/mar/Development/pyFaSTMM";
      flake = false;
    };
    # Same pinned commit as pyFaSTMM's own flake.nix's `fastmm2-src` input.
    fastmm2-src = {
      url = "git+https://bitbucket.org/planetarysystemresearch/fastmm2?rev=4b56dc5b30333f3358e205ba8f88a03ee4d2bb3b";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pymstm-repo,
      mstm-src,
      pyfastmm-repo,
      fastmm2-src,
      ...
    }@inputs:
    let
      inherit (nixpkgs) lib;

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      forEachSupportedSystem =
        f:
        lib.genAttrs supportedSystems (
          system:
          f {
            inherit system;
            pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          }
        );
    in
    {
      packages = forEachSupportedSystem (
        { pkgs, system }:
        {
          # The CLI reference binaries -- same derivations devenv.nix
          # already pulls onto PATH for MstmCliAdapter/Fastmm2CliAdapter,
          # exposed here too so `nix build .#mstm`/`.#fastmm2` work
          # standalone.
          mstm = pkgs.callPackage ./nix/packages/mstm/package.nix { };
          fastmm2 = pkgs.callPackage ./nix/packages/fastmm2/package.nix { };

          # The Python bindings, as real Nix derivations (installable
          # outside a uv-managed venv) -- meson-python drives the exact
          # same meson.build/tools/build_f2py_ext.py that `uv sync`
          # triggers locally, so the compiled extension is the same
          # numpy.f2py output either way. The one thing a plain
          # `src = pymstm-repo` copy doesn't carry over is
          # `external/mstm`'s content: it's a git submodule, and Nix's
          # copy of a local path input only ever sees the outer repo's
          # gitlink entry for it, not the submodule's checked-out files
          # -- so postPatch below repopulates it from the pinned
          # mstm-src input, the same source pyMSTM's own flake.nix
          # already uses for the `mstm`/`mstm-mpi` CLI derivations.
          pymstm = pkgs.python3Packages.buildPythonPackage {
            pname = "pymstm";
            version = "0.1.0";
            pyproject = true;
            src = pymstm-repo;

            postPatch = ''
              rm -rf external/mstm
              cp -r ${mstm-src} external/mstm
              chmod -R u+w external/mstm
            '';

            # nixpkgs' meson-python setup hook pre-runs `meson setup` as
            # its own configurePhase and hands `pypaBuildHook` a
            # `-Cbuild-dir=` pointing at it -- on this nixpkgs/meson-python
            # pairing that pre-configured dir confuses `python -m build`
            # into treating the *build* dir as the source root ("Source
            # .../build does not appear to be a Python project"). Skipping
            # that pre-configure step lets meson-python's own build
            # backend invoke meson itself from the real source root,
            # which is its normal, fully self-contained mode of operation.
            dontUseMesonConfigure = true;

            build-system = [ pkgs.python3Packages.meson-python ];
            nativeBuildInputs = [
              pkgs.meson
              pkgs.ninja
              pkgs.gfortran
              pkgs.gnupatch
            ];
            dependencies = [ pkgs.python3Packages.numpy ];

            # The test suite cross-checks against the standalone `mstm`
            # CLI binary and real MPI runs -- neither is wired up as a
            # build input here (t-bench's own cross-adapter tests already
            # cover that comparison), so skip pytest during the Nix build.
            doCheck = false;
          };

          # Same shape as pymstm above -- FaSTMM2 additionally needs
          # LAPACK/BLAS (plain external DGETRF/DGETRI/... calls, link-time
          # only) and OpenMP (real parallel regions, handled by gfortran's
          # own -fopenmp/-lgomp already baked into
          # tools/build_f2py_ext.py's f2py invocation).
          pyfastmm = pkgs.python3Packages.buildPythonPackage {
            pname = "pyfastmm";
            version = "0.1.0";
            pyproject = true;
            src = pyfastmm-repo;

            postPatch = ''
              rm -rf external/fastmm2
              cp -r ${fastmm2-src} external/fastmm2
              chmod -R u+w external/fastmm2
            '';

            # See pymstm's dontUseMesonConfigure comment above -- same
            # nixpkgs/meson-python interaction, same fix.
            dontUseMesonConfigure = true;

            build-system = [ pkgs.python3Packages.meson-python ];
            nativeBuildInputs = [
              pkgs.meson
              pkgs.ninja
              pkgs.gfortran
              pkgs.gnupatch
            ];
            buildInputs = [
              pkgs.lapack
              pkgs.blas
            ];
            dependencies = [ pkgs.python3Packages.numpy ];

            doCheck = false;
          };
        }
      );

      devShells = forEachSupportedSystem (
        { pkgs, system }:
        {
          default = pkgs.mkShellNoCC {
            packages =
              with pkgs;
              [
                self.formatter.${system}
                gfortran
                lapack
                blas
              ]
              ++ lib.optionals (!pkgs.stdenv.isDarwin) [
                self.packages.${system}.mstm
                self.packages.${system}.fastmm2
              ];
          };
        }
      );

      formatter = forEachSupportedSystem ({ pkgs, ... }: pkgs.nixfmt);
    };
}
