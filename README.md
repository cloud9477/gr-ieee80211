# gr-ieee80211
GNU Radio IEEE 802.11a/g/n/ac transmitter and receiver.

Temp Note
----------
- STF gives trigger of LTF starting point and the CFO updating point by a byte stream.
- LTF if triggered by STF, do auto correlation, give the starting index and residual CFO
- LTF 


FFT support is built in to GNU Radio with FFTW. Here's how it's done. First, define it in your foo_impl.h file. The options are fft_complex_fwd, fft_complex_rev, fft_real_fwd and fft_real_rev.
https://github.com/drmpeg/gr-paint/blob/master/lib/paint_bc_impl.h#L25
https://github.com/drmpeg/gr-paint/blob/master/lib/paint_bc_impl.h#L41

Then initialize it in your foo_impl.cc constructor.

https://github.com/drmpeg/gr-paint/blob/master/lib/paint_bc_impl.cc#L47

Then execute it.

https://github.com/drmpeg/gr-paint/blob/master/lib/paint_bc_impl.cc#L175-L179

You'll need to add the component in the top level CMakeLists.txt.

https://github.com/drmpeg/gr-paint/blob/master/CMakeLists.txt#L78

And link with it in lib/CMakeLists.txt

https://github.com/drmpeg/gr-paint/blob/master/lib/CMakeLists.txt#L25

If you need a window, you can look at the block implementation file for details.

https://github.com/gnuradio/gnuradio/blob/master/gr-fft/lib/fft_v_fftw.cc