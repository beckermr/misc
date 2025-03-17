select
    m.tilename || '-' || m.band as key,
    m.tilename as tilename,
    fai.path as path,
    fai.filename as filename,
    fai.compression as compression,
    m.band as band,
    m.pfw_attempt_id as pfw_attempt_id
from
    prod.proctag t,
    prod.coadd m,
    prod.file_archive_info fai
where
    t.tag='Y6A2_COADD'
    and t.pfw_attempt_id=m.pfw_attempt_id
    and m.filetype='coadd_nobkg'
    and fai.filename=m.filename
    and fai.archive_name='desar2home'; > coadd_data_paths.fits
