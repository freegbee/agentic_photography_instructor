class PrometheusMetrics:
    def __init__(self):
        self.metrics = {}

    def register_metric(self, name, description):
        if name not in self.metrics:
            self.metrics[name] = {
                'description': description,
                'value': 0
            }

    def set_metric(self, name, value):
        if name in self.metrics:
            self.metrics[name]['value'] = value
        else:
            raise ValueError(f"Metric '{name}' not registered.")

    def get_metric(self, name):
        if name in self.metrics:
            return self.metrics[name]['value']
        else:
            raise ValueError(f"Metric '{name}' not registered.")

    def export_metrics(self):
        output = []
        for name, data in self.metrics.items():
            output.append(f"# HELP {name} {data['description']}")
            output.append(f"# TYPE {name} gauge")
            output.append(f"{name} {data['value']}")
        return "\n".join(output)