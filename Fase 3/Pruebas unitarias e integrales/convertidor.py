import nbconvert

# Convertir notebook a script
converter = nbconvert.exporters.ScriptExporter()
body, _ = converter.from_filename("fase2.ipynb")

# Guardar el script
with open("fase2.py", "w", encoding="utf-8") as f:
    f.write(body)


