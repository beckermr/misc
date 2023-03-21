import galsim
from galsim.config.output import OutputBuilder
from galsim.fits import writeMulti
import uuid


class MultiImageModBuilder(OutputBuilder):
    def setup(self, config, base, file_num, logger):
        pass

    def getNFiles(self, config, base, logger=None):
        return 256

    def getFilename(self, config, base, logger):
        uid = uuid.uuid4().hex[0:8]
        return "multi_image_mod_%s.fits" % uid

    def buildImages(
        self, config, base, file_num, image_num, obj_num, ignore, logger
    ):
        images = OutputBuilder.buildImages(
            self, config, base, file_num, image_num, obj_num, ignore, logger)
        return images

    def writeFile(self, data, file_name, config, base, logger):
        writeMulti(data, file_name)


# hooks for the galsim config parser
galsim.config.output.RegisterOutputType(
    'MultiImageMod', MultiImageModBuilder()
)
