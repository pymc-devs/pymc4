import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tqdm import tqdm

__all__ = [
    "trace_scan",
]

def trace_scan(
    loop_fn,
    initial_state,
    elems,
    trace_fn,
    trace_criterion_fn=None,
    static_trace_allocation_size=None,
    parallel_iterations=10,
    name=None,
    debug=False,
):
    with tf.name_scope(name or "trace_scan"), tf1.variable_scope(tf1.get_variable_scope()) as vs:
        if vs.caching_device is None and not tf.executing_eagerly():
            vs.set_caching_device(lambda op: op.device)

        initial_state = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x, name="initial_state"), initial_state
        )
        elems = tf.convert_to_tensor(elems, name="elems")

        length = prefer_static.size0(elems)

        # This is an TensorArray in part because of XLA, which had trouble with
        # non-statically known indices. I.e. elems[i] errored, but
        # elems_array.read(i) worked.
        elems_array = tf.TensorArray(elems.dtype, size=length, element_shape=elems.shape[1:])
        elems_array = elems_array.unstack(elems)

        # Initialize trace arrays.
        dynamic_size, initial_size = True, 0
        if trace_criterion_fn is None:
            dynamic_size, initial_size = tf.is_tensor(length), length
        elif static_trace_allocation_size:
            dynamic_size, initial_size = False, static_trace_allocation_size

        def trace_one_step(state):
            return trace_fn(state)

        def loop_fn_(state, elem):
            return loop_fn(state, elem)

        if debug is True:
            trace_one_step = tf.function(trace_one_step, autograph=False)
            loop_fn_ = tf.function(loop_fn_, autograph=False)

        init_trace = trace_one_step(initial_state)

        def stack(ta, x):
            """
                TODO:
                    we should think how this function
                    can be serialized too. For now
                    the dynamic shape of give tensor
                    is the problem.
            """
            return tf.concat([ta, tf.expand_dims(x, 0)], axis=0)

        trace_arrays = tf.nest.map_structure(
            lambda x: tf.TensorArray(
                x.dtype,  # pylint: disable=g-long-lambda
                size=initial_size,
                dynamic_size=dynamic_size,
                element_shape=x.shape,
            ),
            trace_fn(initial_state),
        )

        def _body(i, state, num_steps_traced, trace_arrays):
            elem = elems_array.read(i)
            state = loop_fn_(state, elem)
            cond = trace_criterion_fn(state) if trace_criterion_fn else True
            if cond is True:
                trace_fn = trace_one_step(state)
                trace_arrays = tf.nest.map_structure(
                    lambda ta, x: ta.write(num_steps_traced, x), trace_arrays, trace_fn,
                )
                num_steps_traced += 1

            return i + 1, state, num_steps_traced, trace_arrays

        num_steps_traced = 0
        i = 0

        if debug is True:
            for i in tqdm(range(length)):
                _, initial_state, num_steps_traced, trace_arrays = _body(
                    i, initial_state, num_steps_traced, trace_arrays
                )
            final_state = initial_state
        else:
            _, final_state, _, trace_arrays = tf.while_loop(
                cond=lambda i, *_: i < length,
                body=_body,
                loop_vars=(0, initial_state, 0, trace_arrays),
                parallel_iterations=parallel_iterations,
            )

        stacked_trace = tf.nest.map_structure(lambda x: x.stack(), trace_arrays)

        # Restore the static length if we know it.
        static_length = tf.TensorShape(None if dynamic_size else initial_size)
        def _merge_static_length(x):
          tensorshape_util.set_shape(x, static_length.concatenate(x.shape[1:]))
          return x

        stacked_trace = tf.nest.map_structure(_merge_static_length, stacked_trace)
        return final_state, stacked_trace
