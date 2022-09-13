import setuptools
import versioneer

if __name__ == "__main__":

    setuptools.setup(
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        package_data={
            'scip': ['utils/logging.yml']
        }
    )
