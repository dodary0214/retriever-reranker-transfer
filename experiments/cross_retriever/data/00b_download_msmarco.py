"""Parallel downloader for the MS MARCO indexes (00_download_indexes.py skips them once present).
The two Lucene indexes are fetched from the rgw.cs.uwaterloo.ca mirror listed by Pyserini (same filename, size and
md5 check as the default source); use it when the default download source is slow or unreachable."""
from pyserini.prebuilt_index_info import IMPACT_INDEX_INFO, TF_INDEX_INFO
from pyserini.util import download_prebuilt_index

RGW = 'https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene/'
for info, n in [(TF_INDEX_INFO, 'msmarco-v1-passage'), (IMPACT_INDEX_INFO, 'msmarco-v1-passage.splade-pp-ed')]:
    info[n]['urls'] = [RGW + info[n]['filename']]
for n in ['msmarco-v1-passage', 'msmarco-v1-passage.splade-pp-ed', 'msmarco-v1-passage.bge-base-en-v1.5']:
    print('==', n, flush=True)
    print(download_prebuilt_index(n, verbose=True), flush=True)
print('ALL_DONE', flush=True)
