with open("notebook.ipynb", "r", encoding="utf-8-sig") as f:
    content = f.read()

with open("notebook_fixed.ipynb", "w", encoding="utf-8") as f:
    f.write(content)