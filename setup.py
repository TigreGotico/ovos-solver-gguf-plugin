#!/usr/bin/env python3
import os

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))

with open(f"{BASEDIR}/README.md", "r") as fh:
    long_desc = fh.read()


def get_version():
    """
    Determine the package version from ovos_gguf_plugin/version.py.
    
    Parses VERSION_MAJOR, VERSION_MINOR, VERSION_BUILD, and VERSION_ALPHA from the file and composes a version string; appends "a{alpha}" when alpha is greater than zero.
    
    Returns:
        version (str): Version string in the form "MAJOR.MINOR.BUILD" or "MAJOR.MINOR.BUILDaALPHA".
    """
    version_file = os.path.join(BASEDIR, 'ovos_gguf_plugin', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if int(alpha):
        version += f"a{alpha}"
    return version


def required(requirements_file):
    """
    Read a requirements file, strip comments and empty lines, and optionally relax version pins.
    
    If the environment variable MYCROFT_LOOSE_REQUIREMENTS is set, version pins using '==' or '~=' are replaced with '>='.
    
    Parameters:
        requirements_file (str): Path to the requirements file relative to BASEDIR.
    
    Returns:
        list[str]: A list of requirement specification strings with comments and blank lines removed.
    """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        if 'MYCROFT_LOOSE_REQUIREMENTS' in os.environ:
            print('USING LOOSE REQUIREMENTS!')
            requirements = [r.replace('==', '>=').replace('~=', '>=') for r in requirements]
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]


CHAT_PLUGIN_ENTRY_POINT = 'ovos-chat-gguf-plugin=ovos_gguf_plugin.chat:GGUFChatEngine'
DIALOG_PLUGIN_ENTRY_POINT = 'ovos-dialog-transformer-gguf-plugin=ovos_gguf_plugin.dialog_transformers:GGUFDialogTransformer'
SUMMARIZER_ENTRY_POINT = 'ovos-summarizer-gguf-plugin=ovos_gguf_plugin.summarizer:GGUFSummarizer'
TRANSLATE_ENTRY_POINT = 'ovos-translate-gguf-plugin=ovos_gguf_plugin.translate:GGUFTextTranslator'
LANG_DETECT_ENTRY_POINT = 'ovos-lang-detect-gguf-plugin=ovos_gguf_plugin.translate:GGUFTextLangDetector'

setup(
    name='ovos-gguf-plugin',
    version=get_version(),
    description='local LLM plugin for OpenVoiceOS persona framework',
    url='https://github.com/TigreGotico/ovos-gguf-plugin',
    author='jarbasai',
    author_email='jarbasai@mailfence.com',
    license='MIT',
    packages=['ovos_gguf_plugin'],
    zip_safe=True,
    keywords='OVOS openvoiceos plugin utterance fallback query',
    entry_points={
        "opm.transformer.dialog": DIALOG_PLUGIN_ENTRY_POINT,
        "opm.agents.chat": CHAT_PLUGIN_ENTRY_POINT,
        "opm.agents.summarizer": SUMMARIZER_ENTRY_POINT,
        "opm.lang.translate": TRANSLATE_ENTRY_POINT,
        "opm.lang.detect": LANG_DETECT_ENTRY_POINT,
    },
    install_requires=required("requirements.txt"),
    long_description=long_desc,
    long_description_content_type='text/markdown'
)