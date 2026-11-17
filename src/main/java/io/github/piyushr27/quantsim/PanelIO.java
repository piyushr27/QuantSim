package io.github.piyushr27.quantsim;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Binary panel format (written by {@code python/export_panel.py}).
 *
 * <pre>
 * dir/shape.txt     "T N"
 * dir/dates.txt     one ISO date per line (T lines)
 * dir/tickers.txt   one ticker per line (N lines)
 * dir/sectors.txt   one sector name per line (N lines)
 * dir/close.f64 ret.f64 adv.f64 vol.f64   little-endian float64, date-major T*N
 * dir/valid.u8      one byte per cell (0/1), date-major
 * </pre>
 * Signals / weights are T*N matrices stored as {@code .f64} or {@code .f32}.
 * Files are memory-mapped and bulk-copied into heap arrays.
 */
public final class PanelIO {
    private PanelIO() {}

    public static Panel load(Path dir) throws IOException {
        String[] shape = Files.readString(dir.resolve("shape.txt")).trim().split("\\s+");
        int T = Integer.parseInt(shape[0]);
        int N = Integer.parseInt(shape[1]);
        List<String> d = Files.readAllLines(dir.resolve("dates.txt"));
        List<String> tk = Files.readAllLines(dir.resolve("tickers.txt"));
        List<String> sc = Files.readAllLines(dir.resolve("sectors.txt"));
        if (d.size() != T || tk.size() != N || sc.size() != N) throw new IOException("metadata does not match shape");
        LocalDate[] dates = d.stream().map(LocalDate::parse).toArray(LocalDate[]::new);
        Map<String, Integer> ids = new HashMap<>();
        int[] sector = new int[N];
        for (int i = 0; i < N; i++) sector[i] = ids.computeIfAbsent(sc.get(i), k -> ids.size());
        double[] close = readMatrix(dir.resolve("close.f64"), T, N);
        double[] ret = readMatrix(dir.resolve("ret.f64"), T, N);
        double[] adv = readMatrix(dir.resolve("adv.f64"), T, N);
        double[] vol = readMatrix(dir.resolve("vol.f64"), T, N);
        boolean[] valid = readMask(dir.resolve("valid.u8"), T, N);
        return new Panel(dates, tk.toArray(String[]::new), sector, close, ret, adv, vol, valid);
    }

    /** Read a T*N matrix; element type is taken from the file extension (.f64 or .f32). */
    public static double[] readMatrix(Path file, int T, int N) throws IOException {
        long cells = (long) T * N;
        boolean f32 = file.toString().endsWith(".f32");
        long bytes = cells * (f32 ? 4 : 8);
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            if (ch.size() != bytes) throw new IOException(file + ": expected " + bytes + " bytes, got " + ch.size());
            MappedByteBuffer mb = ch.map(FileChannel.MapMode.READ_ONLY, 0, bytes);
            mb.order(ByteOrder.LITTLE_ENDIAN);
            double[] out = new double[(int) cells];
            if (f32) {
                float[] tmp = new float[(int) cells];
                mb.asFloatBuffer().get(tmp);
                for (int k = 0; k < tmp.length; k++) out[k] = tmp[k];
            } else {
                mb.asDoubleBuffer().get(out);
            }
            return out;
        }
    }

    public static boolean[] readMask(Path file, int T, int N) throws IOException {
        byte[] b = Files.readAllBytes(file);
        if (b.length != (long) T * N) throw new IOException(file + ": bad size " + b.length);
        boolean[] out = new boolean[b.length];
        for (int k = 0; k < b.length; k++) out[k] = b[k] != 0;
        return out;
    }

    public static void writeMatrix(Path file, double[] data) throws IOException {
        ByteBuffer bb = ByteBuffer.allocate(data.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        bb.asDoubleBuffer().put(data);
        Files.write(file, bb.array());
    }

    public static void save(Panel p, Path dir) throws IOException {
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("shape.txt"), p.T + " " + p.N + "\n");
        Files.write(dir.resolve("dates.txt"), java.util.Arrays.stream(p.dates).map(LocalDate::toString).toList());
        Files.write(dir.resolve("tickers.txt"), List.of(p.tickers));
        Files.write(dir.resolve("sectors.txt"), java.util.Arrays.stream(p.sector).mapToObj(s -> "S" + s).toList());
        writeMatrix(dir.resolve("close.f64"), p.close);
        writeMatrix(dir.resolve("ret.f64"), p.ret);
        writeMatrix(dir.resolve("adv.f64"), p.adv);
        writeMatrix(dir.resolve("vol.f64"), p.vol);
        byte[] v = new byte[p.valid.length];
        for (int k = 0; k < v.length; k++) v[k] = (byte) (p.valid[k] ? 1 : 0);
        Files.write(dir.resolve("valid.u8"), v);
    }
}
