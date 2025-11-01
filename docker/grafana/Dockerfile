FROM grafana/grafana:latest
COPY provisioning /etc/grafana/provisioning

# optional: defaults (admin/pass) können per ENV gesetzt werden
ENV GF_SECURITY_ADMIN_USER=admin
ENV GF_SECURITY_ADMIN_PASSWORD=admin
EXPOSE 3000