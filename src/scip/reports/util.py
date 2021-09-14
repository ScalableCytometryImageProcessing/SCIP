from jinja2 import Environment, FileSystemLoader, select_autoescape


def get_jinja_template(template_dir, template):
    jinja_env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape()
    )
    jinja_template = jinja_env.get_template(template)

    return jinja_template
