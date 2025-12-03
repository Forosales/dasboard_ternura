#!/bin/bash

# =======================================
# AUTO DEPLOY A GITHUB + HEROKU
# =======================================

# Mensaje de commit opcional
MSG=${1:-"Actualización automática del dashboard_ternura"}

echo "---------------------------------------------"
echo "📌 Iniciando deploy de dashboard_ternura..."
echo "---------------------------------------------"

# Moverse a la carpeta donde está tu proyecto
cd "G:/Mi unidad/Consultorias/Signature_product_ternura_WV/dashboard_ternura" || {
    echo "❌ ERROR: No se encontró la carpeta del proyecto"
    exit 1
}

# Agregar cambios
echo "➕ Agregando archivos..."
git add .

# Crear commit
echo "📝 Commit..."
git commit -m "$MSG"

# Subir a GitHub
echo "⏫ Subiendo a GitHub..."
git push origin main

# Subir a Heroku
echo "🚀 Haciendo deploy en Heroku..."
git push heroku main

echo "---------------------------------------------"
echo "✨ DEPLOY COMPLETADO CORRECTAMENTE"
echo "---------------------------------------------"

# Mostrar logs Heroku (opcional)
# heroku logs --tail
