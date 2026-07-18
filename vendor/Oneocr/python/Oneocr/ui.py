"""Python GUI helpers for the unified OneOcr environment.

This module mirrors `OneOcr.Ui.psm1` in PowerShell.  It contains only user
interface helpers so the OCR pipeline itself stays identical between CLI and GUI
entry scripts.
"""

from __future__ import annotations

from .common import AppContext


def get_image_path_from_dialog(context: AppContext) -> str | None:
    """Show a file picker when the GUI script is launched without an image path."""

    import tkinter as tk
    from tkinter import filedialog

    gui_config = context.config['gui']
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title=gui_config['open_dialog_title'],
        filetypes=[
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp'),
            ('All files', '*.*'),
        ],
    )
    root.destroy()
    return file_path or None


def show_result_window(context: AppContext, image_path: str, text: str) -> None:
    """Show a simple OCR result window with copy-to-clipboard support."""

    import tkinter as tk
    from tkinter import scrolledtext

    gui_config = context.config['gui']
    root = tk.Tk()
    root.title(gui_config['window_title'])
    root.geometry('1000x700')

    label = tk.Label(root, text=image_path, anchor='w')
    label.pack(fill='x', padx=10, pady=(10, 5))

    text_box = scrolledtext.ScrolledText(root, wrap='none', font=('Consolas', 10))
    text_box.pack(fill='both', expand=True, padx=10, pady=(0, 10))
    text_box.insert('1.0', text)
    text_box.configure(state='disabled')

    button_frame = tk.Frame(root)
    button_frame.pack(fill='x', padx=10, pady=(0, 10))

    def copy_to_clipboard() -> None:
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()

    copy_button = tk.Button(button_frame, text='Copy to clipboard', command=copy_to_clipboard)
    copy_button.pack(side='left')

    close_button = tk.Button(button_frame, text='Close', command=root.destroy)
    close_button.pack(side='left', padx=(10, 0))

    root.mainloop()


def show_error_dialog(context: AppContext, message: str) -> None:
    """Display an interactive error dialog in GUI mode."""

    import tkinter as tk
    from tkinter import messagebox

    gui_config = context.config['gui']
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    messagebox.showerror(gui_config['window_title'], message)
    root.destroy()
