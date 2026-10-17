"""Dark ttk theme for the Minecraft-Modded2Model window: flat cards, brass accent (Create's colour), no bevels."""
import tkinter as tk
from tkinter import ttk

BG = "#15171c"          # window
CARD = "#1d2027"        # panels
FIELD = "#272b34"       # entries, lists
HOVER = "#2f3440"
BORDER = "#323845"
TEXT = "#e7e9ee"
MUTED = "#8d95a8"
DIM = "#5c6375"
ACCENT = "#e0a040"      # brass
ACCENT_HI = "#f2b85a"
ACCENT_LO = "#b9802c"
ON_ACCENT = "#1b1306"
PREVIEW = "#212226"     # matches the preview renderer's background
LOG = "#101216"

FONT = ("Segoe UI", 10)
SMALL = ("Segoe UI", 9)
BOLD = ("Segoe UI Semibold", 10)
HEAD = ("Segoe UI Semibold", 11)
TITLE = ("Segoe UI Semibold", 20)
MONO = ("Cascadia Mono", 9)


def dark_title_bar(root):
    """Ask Windows 10/11 for a dark title bar to match (ignored elsewhere)."""
    try:
        import ctypes
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        on = ctypes.c_int(1)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (newer, older builds)
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(on), ctypes.sizeof(on)) == 0:
                break
    except Exception:
        pass


def apply(root):
    root.configure(background=BG)
    dark_title_bar(root)
    s = ttk.Style(root)
    s.theme_use("clam")
    flat = dict(bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD)

    s.configure(".", background=BG, foreground=TEXT, font=FONT, fieldbackground=FIELD,
                troughcolor=FIELD, selectbackground=ACCENT, selectforeground=ON_ACCENT,
                insertcolor=TEXT, focuscolor=BG, **flat)
    s.map(".", foreground=[("disabled", DIM)])

    # frames and labels
    s.configure("TFrame", background=BG)
    s.configure("Card.TFrame", background=CARD)
    s.configure("Header.TFrame", background=CARD)
    s.configure("Rule.TFrame", background=BORDER)
    s.configure("TLabel", background=CARD, foreground=TEXT)
    for base in ("Card", "Header"):
        s.configure(f"{base}.TLabel", background=CARD, foreground=TEXT)
    s.configure("Hint.TLabel", background=CARD, foreground=MUTED, font=SMALL)
    s.configure("BgHint.TLabel", background=BG, foreground=MUTED, font=SMALL)
    s.configure("Title.TLabel", background=CARD, foreground=ACCENT, font=TITLE)
    s.configure("Subtitle.TLabel", background=CARD, foreground=MUTED, font=FONT)
    s.configure("Head.TLabel", background=CARD, foreground=TEXT, font=HEAD)
    s.configure("Section.TLabel", background=CARD, foreground=ACCENT, font=BOLD)
    s.configure("Step.TLabel", background=ACCENT, foreground=ON_ACCENT, font=BOLD, padding=(7, 0))
    s.configure("Preview.TLabel", background=PREVIEW, foreground=MUTED)
    s.configure("Status.TLabel", background=BG, foreground=MUTED, font=SMALL)
    s.map("TLabel", foreground=[("disabled", DIM)])

    # buttons
    s.configure("TButton", background=FIELD, foreground=TEXT, padding=(12, 5), relief="flat",
                bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD, focusthickness=0)
    s.map("TButton", background=[("disabled", CARD), ("pressed", BORDER), ("active", HOVER)],
          lightcolor=[("active", HOVER)], darkcolor=[("active", HOVER)],
          foreground=[("disabled", DIM)])
    s.configure("Accent.TButton", background=ACCENT, foreground=ON_ACCENT, font=("Segoe UI Semibold", 11),
                padding=(28, 8), bordercolor=ACCENT_LO, lightcolor=ACCENT, darkcolor=ACCENT)
    s.map("Accent.TButton", background=[("disabled", HOVER), ("pressed", ACCENT_LO), ("active", ACCENT_HI)],
          lightcolor=[("active", ACCENT_HI)], darkcolor=[("active", ACCENT_HI)],
          foreground=[("disabled", DIM)])

    # segmented control (radiobuttons drawn as joined toggle buttons)
    s.configure("Segment.Toolbutton", background=FIELD, foreground=MUTED, padding=(16, 6), font=FONT,
                bordercolor=BORDER, lightcolor=FIELD, darkcolor=FIELD, anchor="center")
    s.map("Segment.Toolbutton",
          background=[("selected", ACCENT), ("active", HOVER)],
          foreground=[("selected", ON_ACCENT), ("active", TEXT)],
          lightcolor=[("selected", ACCENT), ("active", HOVER)],
          darkcolor=[("selected", ACCENT), ("active", HOVER)],
          bordercolor=[("selected", ACCENT_LO)],
          font=[("selected", BOLD)])

    # check and radio buttons
    for w in ("TCheckbutton", "TRadiobutton"):
        s.configure(w, background=CARD, foreground=TEXT, indicatorbackground=FIELD,
                    indicatorforeground=ON_ACCENT, indicatormargin=(0, 0, 8, 0), indicatorsize=15,
                    upperbordercolor=DIM, lowerbordercolor=DIM, padding=(0, 3), focusthickness=0)
        s.map(w, background=[("active", CARD)],
              indicatorbackground=[("disabled", CARD), ("selected", ACCENT), ("active", HOVER)],
              upperbordercolor=[("selected", ACCENT_LO)], lowerbordercolor=[("selected", ACCENT_LO)],
              foreground=[("disabled", DIM)])

    # fields
    field = dict(fieldbackground=FIELD, foreground=TEXT, bordercolor=BORDER, lightcolor=FIELD,
                 darkcolor=FIELD, insertcolor=TEXT, padding=(6, 4), selectbackground=ACCENT,
                 selectforeground=ON_ACCENT)
    s.configure("TEntry", **field)
    s.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", FIELD)],
          fieldbackground=[("disabled", CARD)], foreground=[("disabled", DIM)])
    s.configure("TCombobox", arrowcolor=MUTED, background=FIELD, arrowsize=14, **field)
    s.map("TCombobox", fieldbackground=[("readonly", FIELD), ("disabled", CARD)],
          background=[("active", HOVER), ("readonly", FIELD)], bordercolor=[("focus", ACCENT)],
          foreground=[("disabled", DIM), ("readonly", TEXT)], arrowcolor=[("disabled", DIM), ("active", TEXT)],
          selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)])
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", ON_ACCENT)
    root.option_add("*TCombobox*Listbox.font", FONT)
    root.option_add("*TCombobox*Listbox.borderWidth", 0)

    # notebooks: tabs sit on the window background; the open tab joins its card-coloured page
    s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(0, 4, 0, 0),
                bordercolor=CARD, lightcolor=CARD, darkcolor=CARD)
    s.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(16, 7), font=FONT,
                bordercolor=BG, lightcolor=BG, darkcolor=BG)
    s.map("TNotebook.Tab", background=[("selected", CARD), ("active", HOVER)],
          foreground=[("selected", ACCENT), ("active", TEXT)], font=[("selected", BOLD)],
          lightcolor=[("selected", CARD)], darkcolor=[("selected", CARD)],
          bordercolor=[("selected", CARD)], expand=[("selected", (0, 2, 0, 0))])

    # progress bars, scrollbars, panes, tables
    s.configure("TProgressbar", troughcolor=FIELD, background=ACCENT, bordercolor=FIELD,
                lightcolor=ACCENT, darkcolor=ACCENT, thickness=6)
    s.layout("Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {"sticky": "ns", "children": [
        ("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
    s.configure("Vertical.TScrollbar", troughcolor=FIELD, background=BORDER, bordercolor=FIELD,
                lightcolor=BORDER, darkcolor=BORDER, gripcount=0, arrowsize=10)
    s.map("Vertical.TScrollbar", background=[("active", DIM)], lightcolor=[("active", DIM)],
          darkcolor=[("active", DIM)])
    s.configure("TPanedwindow", background=BG)
    s.configure("Sash", sashthickness=10, gripcount=0, background=BG, lightcolor=BG, darkcolor=BG,
                bordercolor=BG)
    s.configure("Treeview", background=FIELD, fieldbackground=FIELD, foreground=TEXT, rowheight=26,
                bordercolor=FIELD, lightcolor=FIELD, darkcolor=FIELD)
    s.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", ON_ACCENT)])
    s.configure("Treeview.Heading", background=CARD, foreground=MUTED, font=SMALL, relief="flat",
                bordercolor=CARD, lightcolor=CARD, darkcolor=CARD, padding=(6, 4))
    s.map("Treeview.Heading", background=[("active", HOVER)])
    return s


def listbox(parent, **kw):
    """A tk.Listbox dressed to match the theme."""
    return tk.Listbox(parent, background=FIELD, foreground=TEXT, selectbackground=ACCENT,
                      selectforeground=ON_ACCENT, highlightthickness=0, borderwidth=0, relief="flat",
                      font=FONT, activestyle="none", **kw)
