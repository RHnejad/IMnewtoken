"""
lxml_compat.py — Drop-in shim that makes stdlib xml.etree work as lxml.etree.

Provides the subset of lxml API used by smpl_sim:
  - Element, SubElement, ElementTree
  - XMLParser (remove_blank_text ignored)
  - parse (from file or BytesIO)
  - etree.tostring (pretty_print handled via xml.dom.minidom)
  - tree.write (pretty_print kwarg accepted and handled)

Usage:
    Replace `from lxml.etree import ...` with `from lxml_compat import ...`
    or monkey-patch: sys.modules['lxml'] = lxml_compat_module
"""
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom
from io import BytesIO
import sys


class XMLParser:
    """Stub for lxml.etree.XMLParser. remove_blank_text is a no-op."""
    def __init__(self, remove_blank_text=False, **kwargs):
        self._parser = ET.XMLParser()


def parse(source, parser=None):
    """Parse XML from file path or BytesIO, return ElementTree."""
    if isinstance(source, BytesIO):
        return ET.parse(source)
    return ET.parse(source)


def tostring(tree_or_elem, pretty_print=False, encoding=None, **kwargs):
    """Convert element/tree to bytes, optionally pretty-printed."""
    if isinstance(tree_or_elem, ElementTree):
        elem = tree_or_elem.getroot()
    else:
        elem = tree_or_elem

    raw = ET.tostring(elem, encoding="unicode")

    if pretty_print:
        dom = minidom.parseString(raw)
        pretty = dom.toprettyxml(indent="  ")
        # Remove the XML declaration that minidom adds
        lines = pretty.split("\n")
        if lines[0].startswith("<?xml"):
            lines = lines[1:]
        result = "\n".join(lines)
        return result.encode("utf-8") if encoding != "unicode" else result

    return raw.encode("utf-8") if encoding != "unicode" else raw


# Patch ElementTree.write to accept pretty_print kwarg
_original_write = ElementTree.write

def _patched_write(self, file_or_filename, pretty_print=False, **kwargs):
    if pretty_print:
        raw = ET.tostring(self.getroot(), encoding="unicode")
        dom = minidom.parseString(raw)
        pretty = dom.toprettyxml(indent="  ")
        lines = pretty.split("\n")
        if lines[0].startswith("<?xml"):
            lines = lines[1:]
        text = "\n".join(lines)
        if isinstance(file_or_filename, str):
            with open(file_or_filename, "w") as f:
                f.write(text)
        else:
            file_or_filename.write(text.encode("utf-8"))
    else:
        _original_write(self, file_or_filename, **kwargs)

ElementTree.write = _patched_write


# Create a module-like object so `from lxml import etree` works
class _EtreeModule:
    Element = Element
    SubElement = SubElement
    ElementTree = ElementTree
    XMLParser = XMLParser
    parse = staticmethod(parse)
    tostring = staticmethod(tostring)

etree = _EtreeModule()
