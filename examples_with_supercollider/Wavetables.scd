
(
SynthDef(\qTable, { |buf = 0, freq = 250.0, gain = -6.0, out = 0|
	var sig;
	sig = Osc.ar(buf, freq);
	sig = LeakDC.ar(sig);
	sig = LPF.ar(sig, 12000);
	Out.ar(out, sig*gain.dbamp!2);
}).store;
)


s.boot;
s.scope;
s.quit;
