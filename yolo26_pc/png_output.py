"""Headless PNG output using the host's librsvg/Cairo libraries, no pip packages."""
import ctypes as ct
from ctypes.util import find_library
from functools import lru_cache
import os
from pathlib import Path
import struct
import tempfile
import xml.etree.ElementTree as ET
import zlib


class GError(ct.Structure):
    _fields_=[('domain',ct.c_uint),('code',ct.c_int),('message',ct.c_char_p)]


class Rectangle(ct.Structure):
    _fields_=[('x',ct.c_double),('y',ct.c_double),('width',ct.c_double),('height',ct.c_double)]


def bind(library,name,result,*arguments):
    function=getattr(library,name); function.restype=result; function.argtypes=list(arguments)
    return function


@lru_cache(maxsize=1)
def check_renderer():
    libraries=[]
    for name in ('rsvg-2','cairo','gobject-2.0','glib-2.0'):
        path=find_library(name)
        if not path: raise RuntimeError('PNG rendering requires host library '+name+' (used by the GNOME image viewer)')
        libraries.append(ct.CDLL(path))
    rsvg,cairo,gobject,glib=libraries; pointer=ct.c_void_p; error=ct.POINTER(ct.POINTER(GError))
    bind(rsvg,'rsvg_handle_new_from_data',pointer,ct.c_char_p,ct.c_size_t,error)
    bind(rsvg,'rsvg_handle_render_document',ct.c_int,pointer,pointer,ct.POINTER(Rectangle),error)
    bind(cairo,'cairo_image_surface_create',pointer,ct.c_int,ct.c_int,ct.c_int)
    bind(cairo,'cairo_surface_status',ct.c_int,pointer)
    bind(cairo,'cairo_create',pointer,pointer)
    bind(cairo,'cairo_status',ct.c_int,pointer)
    bind(cairo,'cairo_status_to_string',ct.c_char_p,ct.c_int)
    bind(cairo,'cairo_surface_write_to_png',ct.c_int,pointer,ct.c_char_p)
    bind(cairo,'cairo_destroy',None,pointer); bind(cairo,'cairo_surface_destroy',None,pointer)
    bind(gobject,'g_object_unref',None,pointer); bind(glib,'g_error_free',None,ct.POINTER(GError))
    return tuple(libraries)


def png_size(data):
    if (len(data)<33 or data[:16]!=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' or
            zlib.crc32(data[12:29])!=struct.unpack_from('>I',data,29)[0]):
        raise ValueError('Invalid PNG header/CRC')
    return struct.unpack_from('>2I',data,16)


def write_png(svg,path):
    """Render the existing geometry in memory; atomically publish a complete PNG."""
    data=svg.encode('utf-8') if isinstance(svg,str) else svg
    try: root=ET.fromstring(data)
    except ET.ParseError as exc: raise ValueError('Invalid drawing XML: '+str(exc)) from exc
    if root.tag!='{http://www.w3.org/2000/svg}svg': raise ValueError('Expected an SVG document')
    width,height=(int(root.attrib[key]) for key in ('width','height'))
    if min(width,height)<=0 or max(width,height)>16384 or width*height>32*1024*1024:
        raise ValueError('PNG dimensions outside supported image range')
    # SVG 1.1 image links work with the older librsvg shipped by Ubuntu 20.04.
    changed=False
    for image in root.iter('{http://www.w3.org/2000/svg}image'):
        if 'href' in image.attrib:
            image.set('{http://www.w3.org/1999/xlink}href',image.attrib.pop('href')); changed=True
    if changed: data=ET.tostring(root,encoding='utf-8')
    rsvg,cairo,gobject,glib=check_renderer()
    error=ct.POINTER(GError)(); handle=context=surface=None; temporary=None
    def checked_error(message):
        if error:
            message+=': '+error.contents.message.decode('utf-8',errors='replace')
        return ValueError(message)
    def cairo_check(status):
        if status: raise RuntimeError(cairo.cairo_status_to_string(status).decode('utf-8'))
    try:
        handle=rsvg.rsvg_handle_new_from_data(data,len(data),ct.byref(error))
        if not handle: raise checked_error('Cannot load drawing')
        surface=cairo.cairo_image_surface_create(0,width,height)  # ARGB32
        cairo_check(cairo.cairo_surface_status(surface))
        context=cairo.cairo_create(surface); cairo_check(cairo.cairo_status(context))
        viewport=Rectangle(0,0,width,height)
        if not rsvg.rsvg_handle_render_document(handle,context,ct.byref(viewport),ct.byref(error)):
            raise checked_error('Cannot render drawing')
        cairo_check(cairo.cairo_status(context))
        path=Path(path)
        fd,name=tempfile.mkstemp(prefix='.png-',suffix='.tmp',dir=str(path.parent)); os.close(fd)
        temporary=Path(name)
        cairo_check(cairo.cairo_surface_write_to_png(surface,os.fsencode(str(temporary))))
        if png_size(temporary.read_bytes())!=(width,height): raise ValueError('PNG dimensions differ from drawing')
        temporary.replace(path)
        return width,height
    finally:
        if context: cairo.cairo_destroy(context)
        if surface: cairo.cairo_surface_destroy(surface)
        if handle: gobject.g_object_unref(handle)
        if error: glib.g_error_free(error)
        if temporary is not None: temporary.unlink(missing_ok=True)
