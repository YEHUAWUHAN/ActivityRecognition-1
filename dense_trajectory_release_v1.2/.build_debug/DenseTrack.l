-include .build_debug/DenseTrack.d

.build_debug/DenseTrack.l: \
	.build_debug/DenseTrack.d \

debug/DenseTrack: .build_debug/DenseTrack.l \
	.build_debug/DenseTrack.o \

