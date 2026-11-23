FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/fastembed_cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 跳过 Streamlit 首次运行的「输入邮箱」引导，否则容器启动会卡住
RUN mkdir -p /root/.streamlit \
    && printf '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml

COPY rag_v1.py app.py ./

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
