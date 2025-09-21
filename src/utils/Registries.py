from class_registry import ClassRegistry

TRANSFORMER_REGISTRY = ClassRegistry('label')
AGENT_FACTORY_REGISTRY = ClassRegistry('factory_name')

def init_registries():
    """ Initialize the registries by importing the modules that register classes."""
    import transformer
    import transformation_agent
    # The imports above will register all transformers and agent factories