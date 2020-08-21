import e2cnn


g = e2cnn.gspaces.Rot2dOnR2(N = 8)

in_type = e2cnn.nn.FieldType(g, [g.regular_repr])
out_type = e2cnn.nn.FieldType(g, [g.regular_repr])

conv = e2cnn.nn.R2Conv(in_type, out_type, kernel_size=3)
import pdb; pdb.set_trace()
