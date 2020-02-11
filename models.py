from global_vars import keras_models

from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

def getHTMLIDStrFromModelStr(model_str):
        out_str = ''
        layer_reps = [i.split('[')[1] for i in model_str.split(']') if i!='']
        for l in layer_reps:
            out_str += l.replace('.', 'dp')+'___'
        return out_str

class ModelConfig:
    def __init__(self, input_size, model_array=None, model_str=None, actions_size=3):
        self.input_size = input_size # aka shape_size
        self.actions_size = actions_size
        self.model_array = model_array
        self.model_str = model_str

        if model_array is not None and model_str is not None:
            raise ValueError('Please supply only one: model_str or model_array')
        if model_array is not None:
            self.model_array = model_array
            self.model_str = self.getModelStr()
        elif model_str is not None:
            self.model_str = model_str
            self.model_array = self.getModelArray()
        else:
            raise ValueError('You have to supply either a model string or a model array')

        self.checkInputAndOutput()

        self.model = self.makeKerasModel()
        self.html_id_str = getHTMLIDStrFromModelStr(self.model_str)

    def getModelStr(self):
        if self.model_str is not None:
            return self.model_str
        return_str = ''
        for layer_dict in self.model_array:
            layer_str = 'not set yet'
            if layer_dict['layer'] == 'Dense':
                layer_str = 'Dense_'+str(layer_dict['units'])+'_'+layer_dict['activation'][:3]
            elif layer_dict['layer'] == 'Dropout':
                layer_str = 'Drop_'+str(layer_dict['rate'])
            else:
                raise ValueError(layer_dict['layer']+' not recognized.')
            return_str += '['+layer_str+']'
        return return_str

    def getModelArray(self):
        if self.model_array is not None:
            return self.model_array
        raise NotImplemented

    def makeKerasModel(self):
        model = Sequential()

        model.add(Dense(units=self.model_array[0]['units'], activation=self.model_array[0]['activation'], input_shape=(self.input_size,)))

        for layer_dict in self.model_array[1:]:
            if layer_dict['layer'] == 'Dense':
                model.add(Dense(units=layer_dict['units'], activation=layer_dict['activation']))
            elif layer_dict['layer'] == 'Dropout':
                model.add(Dropout(rate=layer_dict['rate']))
            else:
                raise ValueError(layer_dict['layer']+' not recognized.')
        return model

    def checkInputAndOutput(self):
        assert self.model_array[0]['layer'] == 'Dense'
        assert self.model_array[-1]['layer'] == 'Dense'
        assert self.model_array[-1]['activation'] in ['linear', 'softmax']
        assert self.model_array[-1]['units'] == self.actions_size

    def getHTMLIDStr(self):
        out_str = ''
        layer_reps = [i.split('[')[1] for i in self.model_str.split(']') if i!='']
        for l in layer_reps:
            out_str += l.replace('.', 'dp')+'___'
        return out_str
