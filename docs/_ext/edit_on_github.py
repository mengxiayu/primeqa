"""
Sphinx extension to add "Edit on GitHub" links to the
sidebar.
"""

import os
import warnings
import re


__licence__ = 'BSD (3 clause)'


def custom_api_url(project, branch, view, path):
    # redirect the url to link into package to edit readme file and not edit the doc file
    # this script and then action workflow maintains the same information between readme and docs page
    original_url = 'https://github.com/'+project+'/'+view+'/'+branch+'/docs/'+path
    if path != None and 'api/index' in path:
        return 'https://github.com/'+project+'/'+view+'/'+branch+'/docs/api/index.rst'
    elif path != None and 'api/' in path:
        start = 'api/'
        end = '/index'
        str_result = re.search('%s(.*)%s' % (start, end), path)
        if str_result:
            package = str_result.groups()
            return 'https://github.com/'+project+'/'+view+'/'+branch+'/primeqa/'+package[0]+'/README.md'
        else:
            return original_url
    else:
        return original_url


# def get_github_url(app, view, path):
#     return 'https://github.com/{project}/{view}/{branch}/{path}'.format(
#         project=app.config.edit_on_github_project,
#         view=view,
#         branch=app.config.edit_on_github_branch,
#         path=path)


def html_page_context(app, pagename, templatename, context, doctree):
    if templatename != 'page.html':
        return

    if not app.config.edit_on_github_project:
        warnings.warn("edit_on_github_project not specified")
        return

    path = os.path.relpath(doctree.get('source'), app.builder.srcdir)
    show_url = custom_api_url(
        app.config.edit_on_github_project, app.config.edit_on_github_branch, 'blob', path)
    edit_url = custom_api_url(
        app.config.edit_on_github_project, app.config.edit_on_github_branch, 'edit', path)

    # hide edit on packages classes & methods generated by autosummary
    if path != None and '_autosummary' not in path:
        context['show_on_github_url'] = show_url
        context['edit_on_github_url'] = edit_url


def setup(app):
    app.add_config_value('edit_on_github_project', '', True)
    app.add_config_value('edit_on_github_branch', 'master', True)
    app.connect('html-page-context', html_page_context)
