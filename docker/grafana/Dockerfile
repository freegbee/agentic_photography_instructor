FROM grafana/grafana:latest
COPY provisioning /etc/grafana/provisioning
COPY dashboards /var/lib/grafana/dashboards

# Berechtigungen setzen (Grafana UID 472 ist üblich)
USER root
RUN chown -R 472:472 /var/lib/grafana /etc/grafana/provisioning
USER 472

# optional: defaults (admin/pass) können per ENV gesetzt werden
ENV GF_SECURITY_ADMIN_USER=admin
ENV GF_SECURITY_ADMIN_PASSWORD=admin

EXPOSE $MONITORING_GRAFANA_PORT