from thinc.typedefs cimport atom_t

from .stateclass cimport StateClass
from .arc_eager cimport TransitionSystem
from ..vocab cimport Vocab
from ..tokens.doc cimport Doc
from ..structs cimport TokenC
from ._state cimport StateC
from libcpp.list cimport list as cpplist

cdef struct SCORE:
    cpplist[int] action_id
    cpplist[float*] score
    cpplist[int] stack_top_id
    cpplist[int] buffer_first_id

cdef class Parser:
    cdef readonly Vocab vocab
    cdef public object model
    cdef readonly TransitionSystem moves
    cdef readonly object cfg
    cdef public object _multitasks

    cdef SCORE _parseC(self, StateC* state, 
            const float* feat_weights, const float* bias,
            const float* hW, const float* hb,
            int nr_class, int nr_hidden, int nr_feat, int nr_piece) nogil
