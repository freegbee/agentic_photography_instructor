from class_registry import ClassRegistry

TRANSFORMER_REGISTRY = ClassRegistry('label')
AGENT_FACTORY_REGISTRY = ClassRegistry('factory_name')

def init_registries():
    """ Initialize the registries by importing the modules that register classes."""
    init_transformer_registry()
    init_agent_registry()

def init_agent_registry():
    """ Initialize only the agent factory registry by importing the relevant module."""
    import transformation_agent
    # The import above will register all agent factories

def init_transformer_registry():
    """ Initialize only the transformer registry by importing the relevant module."""
    import transformer
    # The import above will register all transformers