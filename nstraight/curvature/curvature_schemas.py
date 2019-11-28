import os
import sys
import inspect
import numpy as np
from pprint import pformat
from collections import OrderedDict
import datajoint as dj
from os.path import dirname as dirname
from skimage.transform import rescale
from inspect import isclass
from itertools import product

dj.config["enable_python_native_blobs"] = True

base_path = dirname(dirname(dirname(inspect.stack()[0][1])))
if base_path not in sys.path:
    sys.path.append(base_path)

from nstraight.data import data_schemas as data
from nstraight.utils.curvature import compute_curvature
from nstraight.utils.utils import key_hash
import nstraight.utils.utils as utils


schema = dj.schema('scadena_neuralstraight_curvature', locals())


@schema
class SpatialRescale(dj.Lookup):
    definition="""
    # Spatial rescaling of the inputs
    rescale_hash            : varchar(32)         # rescale hash
    ---
    rescale_type            : varchar(50)         # type
    """
    
    class Subsample(dj.Part):
        definition="""
        # Rescale by subsampling input
        -> master
        ---
        step                : int              # subsampling step
        """
        _function_name = 'subsample'
        
        @property
        def content(self):
            for p in product([2]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
        
        
    class Interpolate(dj.Part):
        definition="""
        # Rescale by interpolation
        -> master
        ---
        scale               : float            # scale
        """
        _function_name = 'rescale_interpolation'
        
        @property
        def content(self):
            for p in product([0.5]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
        
    class No(dj.Part):
        definition = """
        -> master
        ---
        """
        _function_name = None

        @property
        def content(self):
            yield dict()
            
    def fill(self):
        for rel in [getattr(self, member) for member in dir(self)
                    if isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]:

            for key in rel().content:
                key['rescale_type'] = rel.__name__
                key['rescale_hash'] = key_hash(key)
                if not key in (rel()).proj():
                    self.insert1(key, ignore_extra_fields=True)
                    rel().insert1(key, ignore_extra_fields=True)
        
    def get_scale_func(self, key):
        assert len(self & key) == 1, 'Key must refer to exactly one method'
        rescale_type = (self & key).fetch1('rescale_type')
        
        rel_rescale_type = getattr(self, rescale_type)
        
        if rel_rescale_type._function_name is None:
            rescale_fn = lambda x: x
        else:
            rescale_function = getattr(utils, rel_rescale_type._function_name) 
            params = rel_rescale_type.fetch(as_dict=True)[0]
            rescale_fn = lambda x: rescale_function(x, **params)
        
        return rescale_fn
        
        

@schema
class TemporalFilter(dj.Lookup):
    definition="""
    # Temporal filtering of signals
    filter_hash          : varchar(32)         # temporal filter id
    ---
    filter_type          : varchar(50)         # type
    """    
    class Butterworth(dj.Part):
        definition="""
        # Butterworth filter
        -> master
        ---
        wn                : float               # cutoff frequency
        order             : int                 # order of the filter
        fs                : float               # sampling frequency
        """
        _function_name = 'butter_temporal_filter'
        
        @property
        def content(self):
            for p in product([3], [2, 5], [30]):
                d = dict(zip(self.heading.secondary_attributes, p))
                yield d
            
    class No(dj.Part):
        definition = """
        -> master
        ---
        """
        _function_name = None
        @property
        def content(self):
            yield dict()
            
    def fill(self):
        for rel in [getattr(self, member) for member in dir(self)
                    if isclass(getattr(self, member)) and issubclass(getattr(self, member), dj.Part)]:

            for key in rel().content:
                key['filter_type'] = rel.__name__
                key['filter_hash'] = key_hash(key)
                if not key in (rel()).proj():
                    self.insert1(key, ignore_extra_fields=True)
                    rel().insert1(key, ignore_extra_fields=True)
                    
    
    def get_filter_func(self, key):
        assert len(self & key) == 1, 'Key must refer to exactly one method'
        filter_type = (self & key).fetch1('filter_type')
        
        rel_filter_type = getattr(self, filter_type)
        
        if rel_filter_type._function_name is None:
            filter_fn = lambda x: x
        else:
            filter_function = getattr(utils, rel_filter_type._function_name) 
            params = rel_filter_type.fetch(as_dict=True)[0]
            filter_fn = lambda x: filter_function(x, **params)
        
        return filter_fn
    
                 
                    
@schema
class CurvaturePixels(dj.Computed):
    definition="""
    # Curvature trajectory of movie frames
    -> data.InputResponse.Input
    -> TemporalFilter
    -> SpatialRescale
    ---
    pixel_curvature        : longblob           # curvature trajectory of pixels
    avg_pixel_curvature    : double             # average curvature 
    median_pixel_curvature : double             # median of curvature
    std_pixel_curvature    : double             # standard deviation curvature
    tsteps                 : int                # temporal steps
    spatial_dim            : int                # spatial dimensionality (W x H)
    """
    
    @property
    def key_source(self):
        return super().key_source
    
    
    def make(self, key):
        
        print('Populating ' + pformat(key, indent=10))
        
        # Fetch preprocessing functions:
        temporal_filter = TemporalFilter().get_filter_func(key)
        spatial_rescale = SpatialRescale().get_scale_func(key)
        
        # Fetch data:
        inputs = data.InputResponse.Input().get_inputs(key)
        inputs = temporal_filter(inputs)
        inputs = spatial_rescale(inputs)
        
        # Compute curvature:
        curvature = compute_curvature(inputs)
        
        key['pixel_curvature']        = curvature
        key['avg_pixel_curvature']    = curvature.mean()
        key['median_pixel_curvature'] = np.median(curvature)
        key['std_pixel_curvature']    = curvature.std()
        key['tsteps']                 = inputs.shape[0]
        key['spatial_dim']            = np.prod(inputs.shape[1:])
        self.insert1(key)
        print(key)

                           
    
@schema 
class CurvatureResponse(dj.Computed):
    definition="""
    # Curvature trajectory of neural responses
    -> data.InputResponse.Input
    -> data.BrainArea
    -> data.Layer
    -> TemporalFilter
    ---
    curvature             : longblob           # curvature trajectory of responses
    avg_curvature         : double             # average curvature
    median_curvature      : double             # median of curvature
    std_curvature         : double             # standard deviation curvature
    tsteps                : int                # temporal steps
    num_neurons           : int                # number of neurons
    """
    
    @property
    def key_source(self):
        restr = (data.InputResponse * data.BrainArea * data.Layer).aggr((data.AreaMembership *\
                                                                         data.LayerMembership), n="count(unit_id)")
        
        return data.InputResponse.Input.proj() * restr.proj() * TemporalFilter.proj()
    
    def make(self, key): 
        
        print('Populating ' + pformat(key, indent=10)) 
        
        # Fetch preprocessing functions:
        temporal_filter = TemporalFilter().get_filter_func(key)
        
        # Fetch data
        brain_area = key['brain_area']
        responses  = data.InputResponse.Input().get_responses(key, brain_area)
        responses  = temporal_filter(responses)
        
        # Compute curvature:
        curvature = compute_curvature(responses)
        
        key['curvature']        = curvature
        key['avg_curvature']    = curvature.mean()
        key['median_curvature'] = np.median(curvature)
        key['std_curvature']    = curvature.std()    
        key['tsteps']           = responses.shape[0]
        key['num_neurons']      = responses.shape[1]
        self.insert1(key)
        print(key)



# --------- schemas for subsampled dimensions
        
@schema
class Seed(dj.Lookup):
    definition="""
    # Seeds 
    seed              : int       # seed
    ---
    """
    @property
    def contents(self):
        yield from zip([1,2,3,4,5])
        
        
@schema
class DimensionSample(dj.Lookup):
    definition="""
    # Dimensions to be sample
    sample_size      : int       # size of the sample
    ---
    """
    @property
    def contents(self):
        yield from zip([100, 200, 250, 300, 400, 500, 700])
    

@schema 
class CurvatureResponseSample(dj.Computed):
    definition="""
    # Curvature trajectory of neural responses that are randomly sampled
    -> data.InputResponse.Input
    -> data.BrainArea
    -> data.Layer
    -> TemporalFilter
    -> Seed
    -> DimensionSample
    ---
    curvature             : longblob           # curvature trajectory of responses
    avg_curvature         : float              # average curvature
    median_curvature      : float              # median of curvature
    std_curvature         : float              # standard deviation curvature
    tsteps                : int                # temporal steps
    num_neurons           : int                # number of neurons
    """
    
    @property
    def key_source(self):
        restr = (data.InputResponse * data.BrainArea * data.Layer).aggr((data.AreaMembership *\
                                                                         data.LayerMembership), n="count(unit_id)")
        
        return data.InputResponse.Input.proj() * restr.proj() * TemporalFilter.proj() * Seed * DimensionSample
    
    def make(self, key): 
        
        print('Populating ' + pformat(key, indent=10)) 
        
        sample_size = key['sample_size']
        
        # Fetch preprocessing functions:
        temporal_filter = TemporalFilter().get_filter_func(key)
        
        # Fetch data
        brain_area  = key['brain_area']
        responses   = data.InputResponse.Input().get_responses(key, brain_area)
        num_neurons = responses.shape[1]
        
        # Sample responses
        if num_neurons > sample_size:
            np.random.seed(key['seed'])
            inds_sample = np.random.choice(num_neurons, sample_size, replace = False)
            responses   = responses[:, inds_sample] # sample responses
            
        responses = temporal_filter(responses)
        
        # Compute curvature:
        curvature = compute_curvature(responses).astype(np.float32)
        
        key['curvature']        = curvature
        key['avg_curvature']    = curvature.mean()
        key['median_curvature'] = np.median(curvature)
        key['std_curvature']    = curvature.std()
        key['tsteps']           = responses.shape[0]
        key['num_neurons']      = num_neurons
        self.insert1(key)
        #print(key)
        
def fill():
    TemporalFilter().fill()
    SpatialRescale().fill()