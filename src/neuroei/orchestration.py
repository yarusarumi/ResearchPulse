"""Local Prefect configuration and a narrow draft-only bridge; optional dependency."""
import os
from urllib.parse import urlsplit

from .config import load_config
from .operations import controlled_run


def prepare_local_prefect(config):
    url = os.environ.get('PREFECT_API_URL') or 'http://127.0.0.1:4217/api'
    parsed = urlsplit(url)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1','localhost')
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path != '/api'):
        raise ValueError('This pilot requires a loopback Prefect API; no cloud metadata upload')
    home = config.data_dir / 'prefect'
    home.mkdir(parents=True,exist_ok=True)
    os.environ.update(PREFECT_API_URL=url,PREFECT_HOME=str(home),
        PREFECT_PROFILES_PATH=str(home/'profiles.toml'),PREFECT_SERVER_ANALYTICS_ENABLED='false',
        PREFECT_SERVER_EPHEMERAL_ENABLED='false')
    os.environ.pop('PREFECT_API_KEY',None)
    return {'api_url':url,'home':str(home),'host':parsed.hostname,'port':parsed.port or 80}


def schedule_plan(project,reader_release,read_limit=2):
    return {'project':project,'reader_release':reader_release,'read_limit':read_limit,
            'timezone':'Asia/Shanghai','active':False,'model_retries_at_flow_level':0,
            'concurrency_limit':1,'publication_mode':'draft','feishu_generated_upload_enabled':False,
            'schedules':[{'name':'daily-draft','cron':'0 7 * * *','weekly':False},
                         {'name':'weekly-draft','cron':'0 8 * * 1','weekly':True}]}


def draft_operation(config_path,project,reader_release,read_limit=2,weekly=False):
    config = load_config(config_path)
    prepare_local_prefect(config)
    result = controlled_run(config,project,reader_release=reader_release,read_limit=read_limit,weekly=weekly)
    if result['status'] in ('failed','requires_attention'):
        raise RuntimeError('Controlled draft requires inspection: '+result['attempt_id'])
    return result
