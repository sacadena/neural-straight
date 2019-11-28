import os
import inspect
import numpy as np
from collections import OrderedDict
import datajoint as dj
import pickle
from .datasets import MovieSet


dj.config["enable_python_native_blobs"] = True

PATH = os.path.dirname(os.path.dirname(os.path.dirname(inspect.stack()[0][1])))
DATA_PATH = os.path.join(PATH, 'data/')

ETC_PATH = os.path.join(PATH, 'etc/')

schema = dj.schema('scadena_neuralstraight_movies', locals())


@schema
class BrainArea(dj.Lookup):
    definition = """
    # Brain areas imaged with the mesoscope
    brain_area      : varchar(32)        # short name for cortical area
    ---
    """
    @property
    def contents(self):
        areas = ['A', 'AL', 'AM', 'LI', 'LLA', 'LM', 'MAP', 'P', 'PM', 'POR', 'RL',
       'unknown', 'V1']
        for area in areas:
            d = dict(zip(['brain_area'], [area]))
            yield d

@schema
class Layer(dj.Lookup):
    definition = """
    # Depth layer in cortex
    layer           : varchar(12)       # short name for cortical area
    ---
    """
    @property
    def contents(self):
        layers = ['L1', 'L2/3', 'L4', 'L5', 'L6', 'unset']
        for layer in layers:
            d = dict(zip(['layer'], [layer]))
            yield d

            
@schema
class MovieScan(dj.Manual):
    definition = """
    # Movie scans with mesoscope on multiple brain areas
    animal_id            : int                          # id number
    session              : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    ---
    pipe_version         : smallint                     # id pipeline version
    segmentation_method  : smallint                     # id segmentation method
    spike_method         : smallint                     # id spike extraction method 
    
    """

    class Unit(dj.Part):
        definition = """
        # smaller primary key table for data
        -> master
        unit_id              : int                          # unique per scan & segmentation method
        ---
        pipe_version         : smallint                     # id pipeline version
        segmentation_method  : smallint                     # id segmentation method
        """
        
    def insert_dataset(self, dataset_name, data_path = DATA_PATH):
        
        name       = dataset_name[6:] if dataset_name[:6] == 'movies' else dataset_name
        file_specs = name.split('-')
        file_specs = [int(l) if i < 3 else int(l[-1]) for i,l in enumerate(file_specs)]
        pre_processing = file_specs.pop(3)
        key        = dict(zip(self.heading.attributes, file_specs))
        self.insert1(key, ignore_extra_fields=True)
        
        file_name  = os.path.join(data_path, dataset_name + '.h5')
        dataset    = MovieSet(file_name, 'inputs', 'behavior', 'eye_position', 'responses')
        unit_ids   = dataset.neurons.unit_ids
        brain_areas= dataset.neurons.area
        layers     = dataset.neurons.layer
        tiers      = dataset.tiers
        trial_idxs = dataset.trial_idx
        condition_hashes = dataset.condition_hashes
        
        key['preproc_id'] = pre_processing
        InputResponse.insert1(key, ignore_extra_fields=True)
        
        for unit, area, layer in zip(unit_ids, brain_areas, layers):
            key['unit_id']    = unit
            key['brain_area'] = area
            key['layer']      = layer
            self.Unit().insert1(key, ignore_extra_fields=True)
            AreaMembership().insert1(key, ignore_extra_fields=True)
            LayerMembership().insert1(key, ignore_extra_fields=True)
        
        key.pop('unit_id');
        key.pop('brain_area');
        key.pop('layer');
        
        for trial_idx, condition_hash, tier in zip(trial_idxs, condition_hashes, tiers):
            key['trial_idx'] = trial_idx
            key['condition_hash'] = condition_hash
            key['tier'] = tier
            InputResponse.Input.insert1(key, ignore_extra_fields=True)
            
    def fill(self):
        #self.insert_dataset('movies16314-4-3-pre3-pipe1-seg3-spike5')
        #self.insert_dataset('movies21067-8-9-pre0-pipe1-seg6-spike5')
        self.insert_dataset('movies22084-2-11-pre0-pipe1-seg6-spike5')
        self.insert_dataset('movies22279-5-15-pre0-pipe1-seg6-spike5')
        self.insert_dataset('movies22285-2-10-pre0-pipe1-seg6-spike5')
        

@schema
class AreaMembership(dj.Manual):
    definition ="""
    # area membership of the unit
    -> MovieScan.Unit
    ---
    brain_area           : varchar(256)                 # short name for cortical area
    """

@schema
class LayerMembership(dj.Manual):
    definition ="""
    # layer membership of the unit
    -> MovieScan.Unit
    ---
    layer                : char(12)                     # short name for cortical area
    """

@schema
class InputResponse(dj.Manual):
    definition = """
    # responses of one neuron to the stimulus
    -> MovieScan
    preproc_id           : tinyint                      # preprocessing ID
    ---
    """
    class Input(dj.Part):
        definition = """
        -> master
        condition_hash       : char(20)             # 120-bit hash (The first 20 chars of MD5 in base64)
        trial_idx            : int                  # trial index within sessions
        ---
        tier                 : varchar(32)          # whether the trail was for training, testing or validation
        """
        
        def get_inputs(self, key):
            assert len(self & key) == 1, 'Key must refer to exactly one dataset input trial' 
            file_name = (InputResponse() & key).get_hdf5_filename(key)
            file_path = os.path.join(DATA_PATH, file_name)
            dataset   = MovieSet(file_path, 'inputs')
            key       = (self & key).fetch1('KEY')
            trial_id  = key['trial_idx']
            idx       = np.where(dataset.trial_idx == trial_id)[0][0]
            inputs    = dataset[idx].inputs[0]
            return inputs
        
        def get_responses(self, key, brain_area=None):
            assert len(self & key) == 1, 'Key must refer to exactly one dataset input trial'
            file_name = (InputResponse() & key).get_hdf5_filename(key)
            file_path = os.path.join(DATA_PATH, file_name)
            dataset   = MovieSet(file_path, 'responses')
            key       = (self & key).fetch1('KEY')
            trial_id  = key['trial_idx']
            idx       = np.where(dataset.trial_idx == trial_id)[0][0]
            responses = dataset[idx].responses
            
            if brain_area is None:  
                return responses
            else:
                inds_area = dataset.neurons.area == brain_area
                return responses[:, inds_area]
                

    def get_hdf5_filename(self, key):
        _file_name_format = \
        'movies{animal_id}-{session}-{scan_idx}-pre{preproc_id}-pipe{pipe_version}-seg{segmentation_method}-spike{spike_method}.h5'
        assert len(self & key) == 1, 'Key must refer to exactly one dataset'
        fields = (MovieScan() * self & key).fetch(as_dict=True)[0]
        return _file_name_format.format(**fields)    
    

@schema
class MovieClass(dj.Lookup):
    definition="""
    # Types of movies
    movie_class          : varchar(16)                  # type of movie stimulus
    ---
    """
    @property
    def contents(self):
        movie_classes = ['cinema', 'madmax', 'mei', 'modified', 'monet', 'mousecam',
       'multiobjects', 'object3d', 'opticflow', 'rf', 'unreal',
       'varmaObj', 'youtube']
        for mclass in movie_classes:
            d = dict(zip(['movie_class'], [mclass]))
            yield d
    
@schema
class Movie(dj.Manual):
    definition = """
    # movies used for generating clips and stills
    movie_name           : char(8)                      # short movie title
    ---
    path                 : varchar(255)                 # path at tolias lab of the movie
    movie_class          : varchar(16)                  # type of movie stimulus
    original_file        : varchar(255)                 # name original file
    file_template        : varchar(255)                 # filename template with full path
    file_duration        : float                        # (s) duration of each file (must be equal)
    codec="-c:v libx264 -preset slow -crf 5" : varchar(255)                 # video codec
    movie_description    : varchar(255)                 # full movie title
    frame_rate=30        : float                        # frames per second
    frame_width=256      : int                          # pixels
    frame_height=144     : int                          # pixels
    """
    
    class MovieParams(dj.Part):
        definition ="""
        # Movie parameters for parametric models
        -> master
        ---
        params               : longblob                     # 
        params_file=""       : varchar(255)                 # exported from
        """
        
    ### TODO:
    # re insert the data on a new format friendly to python. 
    
    def insert_names(self):
        with open(os.path.join(ETC_PATH, 'movie_names.pkl'), 'rb') as h:
            movie_names = pickle.load(h)
            self.insert(movie_names)
    def insert_params(self):
        with open(os.path.join(ETC_PATH, 'movie_params.pkl'), 'rb') as h:
            movie_params = pickle.load(h)
            self.MovieParams().insert(movie_params)

    def fill(self):
        self.insert_names()
        self.insert_params()
        
        
@schema
class ConditionClip(dj.Manual):
    definition = """
    # Movie clip condition
    condition_hash       : char(20)                     # 120-bit hash (The first 20 chars of MD5 in base64)
    ---
    movie_name           : char(8)                      # short movie title
    clip_number          : int                          # clip index
    skip_time=0.000      : decimal(7,3)                 # (s) skip to this time in the clip
    cut_after            : decimal(7,3)                 # (s) cut clip if it is longer than this duration
    """
    def insert_conditions(self, conditions_name):
        with open(os.path.join(ETC_PATH, conditions_name), 'rb') as h:
            conditions = pickle.load(h)
            self.insert(conditions, skip_duplicates=True)   
    def fill(self):
        #self.insert_conditions('clips_manolis_movie_names.pkl')
        self.insert_conditions('clips_movie_names.pkl')
        
        


    
    
