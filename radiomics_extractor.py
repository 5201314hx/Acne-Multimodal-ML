from radiomics import featureextractor
import SimpleITK as sitk

def extract_radiomics(image_path, mask_path):
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=32,
        interpolator='sitkBSpline',
        resampledPixelSpacing=[0.1,0.1],
        verbose=False
    )
    img = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    features = extractor.execute(img, mask)
    # Keep only numeric features
    return {k: v for k, v in features.items() if isinstance(v, (int, float))}