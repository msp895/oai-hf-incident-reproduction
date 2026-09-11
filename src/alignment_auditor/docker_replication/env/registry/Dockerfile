FROM python:3.12-slim
COPY server.py /opt/registry/server.py
RUN mkdir -p /srv/registry /var/log
EXPOSE 8080
CMD ["python", "/opt/registry/server.py"]
