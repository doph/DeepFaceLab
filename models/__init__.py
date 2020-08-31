from .ModelBase import ModelBase

def import_model(model_class_name):
    module = __import__('Model_'+model_class_name, globals(), locals(), [], 1)
    return getattr(module, 'Model')

def import_model_tf1(model_class_name):
    module = __import__('Model_'+model_class_name, globals(), locals(), [], 1)
    return getattr(module, 'Model_tf1')