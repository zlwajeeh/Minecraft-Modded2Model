"""Entry point for the Windows executable: opens the classic window, or with arguments runs
the command line (the same options as python -m modded2model)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()  # the block map's worker processes start this same exe
    if len(sys.argv) > 1:
        from modded2model.__main__ import main
        sys.exit(main(sys.argv[1:]))
    from modded2model import gui_classic
    gui_classic.main()
