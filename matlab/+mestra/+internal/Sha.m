classdef Sha
%Sha  SHA-256, written out in MATLAB.
%
%   The digest is written here rather than taken from Java so that the
%   hashing needs nothing outside base MATLAB and behaves the same
%   whether or not a virtual machine is loaded. Checked file replacement
%   uses the JVM separately. Every step here is done on
%   doubles and reduced modulo 2^32, because MATLAB's integer types
%   saturate instead of wrapping.
%
%   The three worked digests of specification section 24 are checked
%   in the test suite.
%
%   See also mestra.supportId.

    methods (Static)

        function hex = hex256(bytes)
        %hex256  The SHA-256 of a byte vector, lower-case hexadecimal.
        %
        %   hex = mestra.internal.Sha.hex256(uint8([0 1 2]))

            bytes = uint8(bytes(:)');
            k = mestra.internal.Sha.constants();
            h = [1779033703 3144134277 1013904242 2773480762 ...
                 1359893119 2600822924 528734635 1541459225];

            % Padding: the byte 0x80, then NUL bytes, then the length
            % in bits as a 64-bit big-endian integer.
            nbits = numel(bytes) * 8;
            padded = [bytes uint8(128)];
            while mod(numel(padded), 64) ~= 56
                padded(end + 1) = 0; %#ok<AGROW>
            end
            lenBytes = zeros(1, 8);
            v = nbits;
            for i = 8:-1:1
                lenBytes(i) = mod(v, 256);
                v = floor(v / 256);
            end
            padded = [padded uint8(lenBytes)];

            two32 = 4294967296;
            for block = 1:64:numel(padded)
                chunk = double(padded(block:block + 63));
                w = zeros(1, 64);
                for i = 1:16
                    j = (i - 1) * 4;
                    w(i) = chunk(j + 1) * 16777216 + chunk(j + 2) * 65536 + ...
                           chunk(j + 3) * 256 + chunk(j + 4);
                end
                for i = 17:64
                    s0 = bitxor(bitxor( ...
                            mestra.internal.Sha.rotr(w(i - 15), 7), ...
                            mestra.internal.Sha.rotr(w(i - 15), 18)), ...
                            floor(w(i - 15) / 8));
                    s1 = bitxor(bitxor( ...
                            mestra.internal.Sha.rotr(w(i - 2), 17), ...
                            mestra.internal.Sha.rotr(w(i - 2), 19)), ...
                            floor(w(i - 2) / 1024));
                    w(i) = mod(w(i - 16) + s0 + w(i - 7) + s1, two32);
                end
                a = h(1); b = h(2); c = h(3); d = h(4);
                e = h(5); f = h(6); g = h(7); hh = h(8);
                for i = 1:64
                    S1 = bitxor(bitxor(mestra.internal.Sha.rotr(e, 6), ...
                                       mestra.internal.Sha.rotr(e, 11)), ...
                                       mestra.internal.Sha.rotr(e, 25));
                    ch = bitxor(bitand(e, f), bitand(two32 - 1 - e, g));
                    t1 = mod(hh + S1 + ch + k(i) + w(i), two32);
                    S0 = bitxor(bitxor(mestra.internal.Sha.rotr(a, 2), ...
                                       mestra.internal.Sha.rotr(a, 13)), ...
                                       mestra.internal.Sha.rotr(a, 22));
                    maj = bitxor(bitxor(bitand(a, b), bitand(a, c)), ...
                                 bitand(b, c));
                    t2 = mod(S0 + maj, two32);
                    hh = g; g = f; f = e;
                    e = mod(d + t1, two32);
                    d = c; c = b; b = a;
                    a = mod(t1 + t2, two32);
                end
                h = mod(h + [a b c d e f g hh], two32);
            end

            hex = '';
            for i = 1:8
                hex = [hex sprintf('%08x', h(i))]; %#ok<AGROW>
            end
        end

        function y = rotr(x, n)
        %rotr  Rotate a 32-bit word right by n places.
            y = bitor(floor(x / 2^n), mod(x * 2^(32 - n), 4294967296));
        end

        function b = int64le(values)
        %int64le  Little-endian bytes of int64 values.
        %   The bytes of the int64 itself, as float64le takes the bytes
        %   of the double: going through a double and back by division
        %   turned -1 into 2^64, which rounds to eight zero bytes, and
        %   would lose any value past 2^53.
            b = typecast(int64(values(:)'), 'uint8');
            if ~mestra.internal.Sha.littleEndian()
                b = reshape(flipud(reshape(b, 8, [])), 1, []);
            end
        end

        function b = float64le(values)
        %float64le  Little-endian bytes of float64 values.
            b = typecast(double(values(:)'), 'uint8');
            if ~mestra.internal.Sha.littleEndian()
                b = reshape(flipud(reshape(b, 8, [])), 1, []);
            end
        end

        function tf = littleEndian()
        %littleEndian  True on a little-endian machine.
            tf = typecast(uint16(1), 'uint8');
            tf = tf(1) == 1;
        end

        function k = constants()
        %constants  The 64 round constants of SHA-256.
            k = [1116352408 1899447441 3049323471 3921009573 961987163 ...
                 1508970993 2453635748 2870763221 3624381080 310598401 ...
                 607225278 1426881987 1925078388 2162078206 2614888103 ...
                 3248222580 3835390401 4022224774 264347078 604807628 ...
                 770255983 1249150122 1555081692 1996064986 2554220882 ...
                 2821834349 2952996808 3210313671 3336571891 3584528711 ...
                 113926993 338241895 666307205 773529912 1294757372 ...
                 1396182291 1695183700 1986661051 2177026350 2456956037 ...
                 2730485921 2820302411 3259730800 3345764771 3516065817 ...
                 3600352804 4094571909 275423344 430227734 506948616 ...
                 659060556 883997877 958139571 1322822218 1537002063 ...
                 1747873779 1955562222 2024104815 2227730452 2361852424 ...
                 2428436474 2756734187 3204031479 3329325298];
        end
    end
end
