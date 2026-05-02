from yamnet_cry_distill_int8.student.dscnn import build_student
from yamnet_cry_distill_int8.teacher import MEL_BINS, NUM_CLASSES, PATCH_FRAMES


def test_student_under_100k_params():
    model = build_student()
    n = model.count_params()
    assert n <= 100_000, f"student has {n} params, budget is 100K"


def test_student_io_shape():
    model = build_student()
    assert model.input_shape == (None, PATCH_FRAMES, MEL_BINS, 1)
    assert model.output_shape == (None, NUM_CLASSES)
