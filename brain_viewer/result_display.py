"""Display-only options; imported measurements are retained unchanged."""
import numpy as np

COLORMAPS=('auto','inferno','turbo','viridis','coolwarm','jet')


def finite_option(value,default,lo,hi):
    try:value=float(value)
    except (ValueError,TypeError):return default
    return float(np.clip(value,lo,hi)) if np.isfinite(value) else default


def metric_display(result,metric):
    settings=result.settings.get('display',{}).get(result.metrics[metric],{}) if result else {}
    palette=settings.get('colormap','auto')
    mode=settings.get('threshold_mode','none')
    return {'colormap':palette if palette in COLORMAPS else 'auto',
            'reverse':bool(settings.get('reverse',False)),
            'threshold_mode':mode if mode in ('none','above','absolute') else 'none',
            'threshold':finite_option(settings.get('threshold',0),0,-1e12,1e12)}


def marker_options(result):
    settings=result.settings.get('markers',{}) if result else {}
    return {'radius_mm':finite_option(settings.get('radius_mm',2),2,.5,8),
            'opacity':finite_option(settings.get('opacity',1),1,0,1),
            'slab_mm':finite_option(settings.get('slab_mm',4),4,.5,30),
            'labels':bool(settings.get('labels',False))}


def spatial_options(result):
    settings=result.settings.get('spatial',{}) if result else {}
    mode=settings.get('mode','points')
    return {'mode':mode if mode in ('points','surface','both') else 'points',
            'distance_mm':finite_option(settings.get('distance_mm',15),15,1,50),
            'method':'idw_inverse_square','neighbors':4,'distance_metric':'euclidean_mm',
            'threshold_after_interpolation':True,'recording_range_estimate':False}


def visible_values(values,options):
    valid=np.isfinite(values)
    mode=options['threshold_mode'];threshold=options['threshold']
    if mode=='above':valid &= values>=threshold
    elif mode=='absolute':valid &= np.abs(values)>=abs(threshold)
    return valid
