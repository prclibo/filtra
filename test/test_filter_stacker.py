import e2cnn

from sscnn.filter_stacker import SingleBlockFilterStacker


g = e2cnn.gspaces.Rot2dOnR2(N=8)

stacker = SingleBlockFilterStacker(g.trivial_repr, g.regular_repr, k_size=3)

in_type = e2cnn.nn.FieldType(g, [g.trivial_repr] * 5)
out_type = e2cnn.nn.FieldType(g, [g.regular] * 5)

