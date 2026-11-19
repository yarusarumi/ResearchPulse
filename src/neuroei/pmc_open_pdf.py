"""Resolve exact DOI to a licensed published PDF in the NLM PMC Cloud Service."""

import json
import re
from urllib.parse import urlsplit

from .sources import normalize_doi, paper_title_key

ID_CONVERTER = 'https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/'
BUCKET = 'https://pmc-oa-opendata.s3.amazonaws.com'
LICENSES = {'CC BY', 'CC0'}


def supports_open_pdf(paper):
    doi = paper.get('doi') or ''
    # The archive PDF has its own verified identity, including for MDPI papers.
    return bool(re.fullmatch(r'10\.\d{4,9}/\S+', doi) and paper['paper_id'] == 'doi:'+doi)


def _json(client, url, *, types=('application/json',), **kwargs):
    with client.stream('GET', url, **kwargs) as response:
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').split(';', 1)[0].strip().lower()
        if response.status_code != 200 or content_type not in types:
            raise ValueError('PMC metadata response is not JSON')
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > 1_000_000:
                raise ValueError('PMC metadata exceeds byte budget')
    return json.loads(data)


def verify_pdf_metadata(paper, metadata):
    pmcid, version = metadata.get('pmcid'), metadata.get('version')
    if (not re.fullmatch(r'PMC[1-9]\d*', str(pmcid)) or type(version) is not int or version < 1
            or normalize_doi(metadata.get('doi')) != paper['doi'] or not paper_title_key(paper['title'])
            or paper_title_key(metadata.get('title')) != paper_title_key(paper['title'])
            or metadata.get('is_pmc_openaccess') is not True or metadata.get('is_manuscript') is not False
            or metadata.get('is_historical_ocr') is not False or metadata.get('is_retracted') is not False
            or metadata.get('license_code') not in LICENSES):
        raise ValueError('PMC published open-access PDF identity or license unverified')
    item = f'{pmcid}.{version}'
    source = urlsplit(metadata.get('pdf_url') or '')
    checksum = re.fullmatch(r'md5=([0-9a-f]{32})', source.query)
    if (source.scheme != 's3' or source.netloc != 'pmc-oa-opendata'
            or source.path != f'/{item}/{item}.pdf' or source.fragment or not checksum):
        raise ValueError('PMC PDF location or checksum unverified')
    return BUCKET+source.path, checksum[1]


def lookup_pdf(client, paper):
    doi = paper['doi']
    payload = _json(client, ID_CONVERTER, params={'ids': doi, 'format': 'json', 'versions': 'yes'})
    records = payload.get('records') or []
    if payload.get('status') != 'ok' or len(records) != 1:
        raise ValueError('Ambiguous PMC DOI lookup')
    record, = records
    if record.get('status') == 'error' or normalize_doi(record.get('doi')) != doi or not record.get('pmcid'):
        return None
    versions = [v['pmcid'] for v in record.get('versions', []) if v.get('current') is True
                and normalize_doi(v.get('doi')) == doi and 'pmcid' in v]
    if (len(versions) != 1 or not re.fullmatch(r'PMC[1-9]\d*', record['pmcid'])
            or not re.fullmatch(re.escape(record['pmcid'])+r'\.[1-9]\d*', versions[0])):
        raise ValueError('Ambiguous PMC dataset version')
    item, = versions
    metadata_url = f'{BUCKET}/{item}/{item}.json'
    metadata = _json(client, metadata_url, types=('application/json', 'binary/octet-stream', 'application/octet-stream'))
    url, checksum = verify_pdf_metadata(paper, metadata)
    if item != f"{metadata['pmcid']}.{metadata['version']}":
        raise ValueError('PMC metadata version differs from DOI resolver')
    fields = ('pmcid', 'version', 'doi', 'title', 'is_pmc_openaccess', 'is_manuscript',
              'is_historical_ocr', 'is_retracted', 'license_code', 'pdf_url')
    return url, checksum, metadata_url, {field: metadata[field] for field in fields}
