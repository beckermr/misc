import galsim
from galsim.config.output import OutputBuilder
from galsim.fits import writeMulti


class MultiImageModBuilder(OutputBuilder):
    def getFilename(self, config, base, logger):
        return "multi_image_mod_%d.fits" % base["file_num"]

    def writeFile(self, data, file_name, config, base, logger):
        writeMulti(data, file_name)


# hooks for the galsim config parser
galsim.config.output.RegisterOutputType(
    'MultiImageMod', MultiImageModBuilder()
)
