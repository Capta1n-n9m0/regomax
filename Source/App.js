const fs = require("fs");
const fsp = require("fs").promises;
const path = require("path");
const bsplit = require("buffer-split");
const Network = require("./Network");
const Matrix = require("./Matrix");
const {format} = require('util');
const printf = require("printf");

const eps_pagerank = 1e-13;


/**
 * @return {number}
 */
function getTime() {
    return (new Date()).getTime();
}

/**
 * @param {Matrix} a
 * @param {string} filename
 * @param {string} node_file_names
 */
function print_mat(a, filename, node_file_names = null) {
    let i, j, dimx, dimy, len = 0, nlen, l;
    let node_names = [];
    if (node_file_names !== null) {
        const data = fs.readFileSync(processFilename(node_file_names));
        const lines = bsplit(data, Buffer.from("\n"));
        for (const line of lines) {
            if (line.length === 0) continue;
            node_names.push(line.toString());
        }
    }
    dimx = a.x;
    dimy = a.y;
    let fp = fs.openSync(processFilename(filename, "Results"), 'w');
    let buffer = "";
    for (let i = 0; i < dimy; i++) {
        for (let j = 0; j < dimx; j++) {
            buffer += printf("%5d\t  %5d\t  %24.26f", i, j, a.mat[i][j]);
            if (i < len && j < len) {
                buffer += printf("\t%s\t", node_names[i], node_file_names[j]);
            }
            buffer += "\n";
        }
        buffer += "\n";
        fs.writeSync(fp, buffer);
        buffer = "";
    }
    fs.writeSync(fp, buffer);
    fs.closeSync(fp);
}

/**
 * @param {float[]} a
 * @return float
 */
function norm1(a) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += Math.abs(a[i]);
    return sum;
}

/**
 * @param {float[]} t
 * @param {float} sp
 * @param {float[]} a
 */
function lam_diff(t, sp, a) {
    if (t.length !== a.length) throw "dimensions error";
    let i;
    for (i = 0; i < t.length; i++) t[i] -= sp * a[i];
}

/**
 * @param {float[]} right
 * @param {float[]} left
 * @param {float[]} v
 */
function projectQ(right, left, v) {
    let sp;
    sp = scalar_product(left, v);
    lam_diff(v, sp, right);
}

/**
 * @param {float[]} right
 * @param {float[]} left
 * @param {float[]} v
 * @param {float} f
 */
function projectP(right, left, v, f = 1) {
    let i, n = v.length;
    let sp;
    sp = scalar_product(left, v) / f;
    if (v.length !== right.length) throw "dimensions error";
    for (i = 0; i < n; i++) v[i] = sp * right[i];
}

/**
 * @param {string} filename
 * @param {string} folder
 * @return string
 */
function processFilename(filename, folder = "Data") {
    return path.join(__dirname, "..", folder, filename);
}

/**
 * @param {float[]} a
 * @param {float[]} b
 * @return float
 */
function scalar_product(a, b) {
    if (a.length !== b.length) throw "dimensions error";
    let sum;
    let i;
    sum = 0;
    for (i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}

/**
 * @param {float[]} a
 * @param {float[]} b
 * @return float
 */
function diff_norm_rel(a, b) {
    let sum, ss;
    let i, n;
    n = a.length;
    sum = 0.0;
    //#pragma omp parallel for reduction(+:sum)
    for (i = 0; i < n; i++) {
        ss = Math.abs(a[i]) + Math.abs(b[i]);
        if (ss === 0) continue;
        sum += Math.abs(a[i] - b[i]) / ss;
    }
    return sum;
}

/**
 *
 * @param {float[]}a
 * @param {float[]}b
 * @return float
 */
function diff_norm1(a, b) {
    let sum;
    if (a.length !== b.length) throw "dimention error";
    sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += Math.abs(a[i] - b[i]);
    }
    return sum;
}

/**
 * @param{any[]} a
 * @param{any[]} b
 */
function swap(a, b) {
    let t1 = [], t2 = [];
    for (const e of a) {
        t1.push(e);
    }
    for (const e of b) {
        t2.push(e);
    }
    a.length = 0;
    b.length = 0;
    for (const e of t1) {
        b.push(e);
    }
    for (const e of t2) {
        a.push(e);
    }
}

/**
 * @param {float[]} a
 * @return float
 */
function sum_vector(a) {
    let sum;
    let i, n;
    n = a.length;
    sum = 0.0;
    for (i = 0; i < n; i++) sum += a[i];
    return sum;
}

/**
 * @param {float[]} a
 * @return float
 */
function pagerank_normalize(a) {
    let sum;
    sum = sum_vector(a);
    for (let i = 0; i < a.length; i++) {
        a[i] /= sum;
    }
    return sum;
}

/**
 * @param{float[]} pagerank
 * @param{Network} net
 * @param{float} delta_alpha
 * @param{number} iprint
 * @param{number[]} node
 * @param{number} trans_frag
 * @return Promise<float>
 */
async function calc_pagerank_project(pagerank, net, delta_alpha, iprint, node, trans_frag) {
    let quality, quality_rel, q1, qfak, pnorm, dlambda, dlambda_old;
    let i, max_iter, l;
    console.log("calc_pagerank_project()");

    if (iprint <= 0) iprint = 1;
    max_iter = Math.floor(-Math.log(eps_pagerank) / (delta_alpha + 3E-7));
    max_iter *= 2;

    console.log(printf("max_iter = %d", max_iter));
    qfak = 1.0 + delta_alpha / 2.0;
    pnorm = pagerank_normalize(pagerank);
    let a = [];
    for (const e of pagerank) a.push(e);
    quality_rel = 1e40;
    dlambda = 0;
    for (l = 0; l < node.length; l++) {
        dlambda += pagerank[node[l]];
        pagerank[node[l]] = 0;
    }
    dlambda_old = dlambda;
    pnorm = pagerank_normalize(pagerank);
    if (trans_frag) dlambda = 1.0 - pnorm;
    for (i = 0; i <= max_iter; i++) {
        swap(a, pagerank);
        if (trans_frag) {
            net.GTmult(delta_alpha, pagerank, a);
        } else {
            net.GGmult(delta_alpha, pagerank, a);
        }
        //pnorm = pagerank_normalize(pagerank);
        //console.log(printf("--> %5d  %25.16f", i, pnorm));
        dlambda = 0;
        for (l = 0; l < node.length; l++) {
            dlambda += pagerank[node[l]];
            pagerank[node[l]] = 0;
        }
        pnorm = pagerank_normalize(pagerank);
        if (trans_frag) dlambda = 1.0 - pnorm;

        if (i % iprint === 0 || i === max_iter) {
            quality = diff_norm1(pagerank, a);
            q1 = quality_rel;
            quality_rel = diff_norm_rel(pagerank, a);
            //      pnorm=pagerank_normalize(pagerank);
            //      pnorm=sum_vector(pagerank);
// #pragma omp critical(print)
            // {
            console.log(printf("%5d  %18.10f  %18.10f  %25.16f  %18.10f  %25.16f", i, quality, quality_rel, dlambda, Math.abs(dlambda - dlambda_old), pnorm));
            //     fflush(stdout);
            // }
            dlambda_old = dlambda;
            if (quality_rel < eps_pagerank) break;
            if (quality_rel < 1e-3) {
                if (quality_rel * qfak > q1) break;
            }
        }
    }
// #pragma omp critical(print)
    // {
    //     printf("Convergence at i = %d  with lambda = %25.16lg.\n", i, 1.0 - dlambda);
    console.log(printf("Convergence at i = %d  with lambda = %25.16f.\n", i, 1.0 - dlambda));
    //     fflush(stdout);
    // }
    return dlambda;
}

/**
 * @param{float[]} right
 * @param{float[]} left
 * @param{float[]} pg
 * @param{Network} net
 * @param{float} delta_alpha
 * @param{number[]} node
 * @return Promise<float>
 */
async function compute_project(right, left, pg, net, delta_alpha, node) {
    const iprint = 10;
    console.log("compute_project()");
    let sp, dlambda1, dlambda2, dlambda3;
    let node0 = [];

    right = right.map(() => {
        return 1.0
    });
    left = left.map(() => {
        return 1.0
    });
    pg = pg.map(() => {
        return 1.0
    });

// #pragma omp parallel sections
    {

// #pragma omp section
        let p2 = calc_pagerank_project(left, net, delta_alpha, iprint, node, 1);
// #pragma omp section
        let p1 = calc_pagerank_project(right, net, delta_alpha, iprint, node);
// #pragma omp section
        let p3 = calc_pagerank_project(pg, net, delta_alpha, iprint, node0);
        dlambda1 = await p1;
        dlambda2 = await p2;
        dlambda3 = await p3;

    }

    sp = 1.0 / scalar_product(left, right);
    for (let i = 0; i < left.length; i++) left[i] = left[i] * sp;
    sp = scalar_product(left, right);
    // #pragma omp critical(print)
    // {
    //     printf("dlambda = %24.16lg   diff = %lg\n",
    //         dlambda1, abs(dlambda1 - dlambda2));
    //     printf("TEST: psi_left^T * psi_right = %26.16lg\n", sp);
    console.log(printf("dlambda = %24.16f   diff = %f\n", dlambda1, Math.abs(dlambda1 - dlambda2)));
    console.log(printf("TEST: psi_left^T * psi_right = %26.16f\n", sp));
    //     fflush(stdout);
    // }

    return dlambda1;
}

/**
 * @param{Matrix} G_R
 * @param{Matrix} G_rr
 * @param{Matrix} G_pr
 * @param{Matrix} G_qr
 * @param{Matrix} G_I
 * @param{float[]} psiL
 * @param{float[]} psiR
 * @param{float[]} pg
 * @param{Network} net
 * @param{float} delta_alpha
 * @param{number[]} node
 */
async function compute_GR(G_R, G_rr, G_pr,
                          G_qr, G_I, psiL,
                          psiR, pg, net,
                          delta_alpha, node) {
    let n = net.size;
    let nr = node.length;
    let ns = n - nr;
    if (G_R.x !== nr || G_R.y !== nr) throw "Wrong matrix size of G_R  in compute_GR";
    if (G_rr.x !== nr || G_rr.y !== nr) throw "Wrong matrix size of G_rr  in compute_GR";
    if (G_pr.x !== nr || G_pr.y !== nr) throw "Wrong matrix size of G_pr  in compute_GR";
    if (G_qr.x !== nr || G_qr.y !== nr) throw "Wrong matrix size of G_qr  in compute_GR";
    if (G_I.x !== nr || G_I.y !== nr) throw "Wrong matrix size of G_I  in compute_GR";
    let dlambda;

    let j, l;
    let quality;

    let i, max_iter;

    max_iter = Math.floor(-Math.log(eps_pagerank) / (delta_alpha + 3e-7));
    max_iter *= 2;

    // printf("Computation of left and right eigenvectors of G_ss\n");
    // fflush(stdout);
    console.log("Computation of left and right eigenvectors of G_ss");

    dlambda = await compute_project(psiR, psiL, pg, net, delta_alpha, node);

    let input, output, s, t, f, f2;
    input = [];
    output = [];
    s = [];
    t = [];
    f = [];
    f2 = [];
    for (let i = 0; i < n; i++) {
        input.push(0);
        output.push(0);
        s.push(0);
        t.push(0);
        f.push(0);
        f2.push(0);
    }
    // #pragma omp parallel for schedule(dynamic) private(in, out, s, t, f, f2, j, l, quality)
    for (i = 0; i < nr; i++) {
        // in.put_value(0.0);
        input[node[i]] = 1;
        net.GGmult(delta_alpha, output, input);
        input[node[i]] = 0;
        for (j = 0; j < nr; j++) {
            G_R.mat[j][i] = output[node[j]];
            G_rr.mat[j][i] = output[node[j]];
            output[node[j]] = 0;
        }
        // s = output;
        for (let i = 0; i < output.length; i++) s[i] = output[i];
        projectP(psiR, psiL, output, dlambda);
        projectQ(psiR, psiL, s);
        // f = s;
        for (let i = 0; i < s.length; i++) f[i] = s[i];

        for (l = 0; l < max_iter; l++) {
            t = new Array(s.length); for(let i = 0; i < s.length; i++){t[i] = s[i]}
            net.GGmult(delta_alpha, f2, f, 0);
            swap(f, f2);
            for (j = 0; j < nr; j++) f[node[j]] = 0;
            projectQ(psiR, psiL, f);
            // s += f;
            if (s.length !== f.length) throw "dimensions error";
            for (let i = 0; i < s.length; i++) s[i] += f[i];
            quality = diff_norm1(t, s);
            // #pragma omp critical(print)
            // {
            if (l % 10 === 0) {
                //         printf("%5d  %5d  %18.10lg  %18.10lg\n", i, l, quality, norm1(f));
                console.log(printf("%5d  %5d  %18.10f  %18.10f", i, l, quality, norm1(f)));
                //         fflush(stdout);
            }
            // }
            if (quality <= 0) break;
        }
// #pragma omp critical(print)
        // {
        //     printf("%5d  ", i);
        //     printf("Convergence: %5d  %5d  %18.10lg  %18.10lg\n",
        console.log(printf("%5d  Convergence: %5d  %5d  %18.10f  %18.10f\n", i, i, l, quality, norm1(f)));
        //         i, l, quality, norm1(f));
        //     fflush(stdout);
        // }
        net.GGmult(delta_alpha, f, output, 0);
        for (j = 0; j < nr; j++) {
            G_pr.mat[j][i] = f[node[j]];
        }
        net.GGmult(delta_alpha, f, s, 0);
        for (j = 0; j < nr; j++) {
            G_qr.mat[j][i] = f[node[j]];
        }
        // out += s;
        if (output.length !== s.length) throw "dimensions error";
        for (let i = 0; i < output.length; i++) output[i] += s[i];
        net.GGmult(delta_alpha, f, output, 0);
        for (j = 0; j < nr; j++) {
            G_I.mat[j][i] = f[node[j]];
            G_R.mat[j][i] += f[node[j]];
        }
    }
}

async function main(argv) {
    const start = getTime();
    const netfile = argv[2];
    const delta_alpha = parseFloat(argv[3]);
    const iprint = parseInt(argv[4]);
    const print_number = parseInt(argv[5]);
    const ten_number = parseInt(argv[6]);
    let nodefile = argv[7];
    const nodefilenames = argv[8];
    console.log(printf("input-file, 1-alpha, iprint, print_number, ten_number  = %s  %f  %d  %d  %d\n", netfile, delta_alpha, iprint, print_number, ten_number));
    console.log(printf("nodefile = %s\n", nodefile));
    console.log(printf("file of node names = %s\n", nodefilenames));

    let node = [];
    const len = await fsp.readFile(processFilename(nodefile))
        .then(data => {
            const lines = bsplit(data, Buffer.from("\n"));
            let len = parseInt(lines[0].toString());
            for (const line of lines.slice(1)) {
                if (line.length === 0) continue;
                node.push(parseInt(line.toString()));
            }
            return len;
        });
    console.log(printf("reading of nodefile finished: len = %d\n", len));
    const net = new Network(netfile);
    const GR = new Matrix(len, len);
    const Grr = new Matrix(len, len);
    const Gpr = new Matrix(len, len);
    const Gqr = new Matrix(len, len);
    const GI = new Matrix(len, len);
    const n = net.size;
    const psiL = [], psiR = [], pg = [], a = [], small_pg = [], b = [];
    for (let i = 0; i < n; i++) {
        psiL.push(0);
        psiR.push(0);
        pg.push(0);
        a.push(0);
    }
    for (let i = 0; i < len; i++) {
        small_pg.push(1);
        b.push(0);
    }
    nodefile = nodefile.split(".")[0];
    await compute_GR(GR, Grr, Gpr, Gqr, GI, psiL, psiR, pg, net, delta_alpha, node);
    print_mat(Gqr, `Gqr_${net.base_name}_${nodefile}_${len}.dat`, nodefilenames);
    console.log(`Calculations took ${(getTime() - start) / 1000} sec\n`);
}

module.exports = main;
