FROM alpine:3.18
ARG PROM_VERSION=2.45.0
RUN apk add --no-cache ca-certificates curl tar gettext
# Download Prometheus binary
RUN curl -fsSL "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz" \
  -o /tmp/prom.tar.gz \
  && tar -xzf /tmp/prom.tar.gz -C /tmp \
  && mv /tmp/prometheus-${PROM_VERSION}.linux-amd64/prometheus /bin/prometheus \
  && mv /tmp/prometheus-${PROM_VERSION}.linux-amd64/promtool /bin/promtool \
  && chmod +x /bin/prometheus /bin/promtool \
  && rm -rf /tmp/*
WORKDIR /etc/prometheus
COPY prometheus.yml.template /etc/prometheus/prometheus.yml.template
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
VOLUME ["/prometheus"]
EXPOSE 9090
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]