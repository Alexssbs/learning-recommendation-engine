# Usar una imagen oficial y ligera de Python
FROM python:3.13-slim

# Copiar el binario de uv directamente (super rápido)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Configuraciones recomendadas para uv en Docker
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Establecer directorio de trabajo
WORKDIR /app

# Copiar los archivos de dependencias
COPY pyproject.toml uv.lock ./

# Instalar dependencias usando uv
RUN uv sync --frozen --no-install-project --no-dev

# Copiar todo el código fuente, incluidos los modelos
COPY . .

# Comando para iniciar la API en producción
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "10000"]
