// noinspection DuplicatedCode

const fsp = require("fs").promises;
const path = require("path");
const bsplit = require("buffer-split");
const Network = require("./Network");
const Matrix = require("./Matrix");
const printf = require("printf");
const Vector = require("./Vector");
const {Worker} = require("worker_threads");

const eps_pagerank = 1e-13;


/**
 * @return {number}
 */
function getTime() {
    return (new Date()).getTime();
}

/**
 * @param {Vector} right
 * @param {Vector} left
 * @param {Vector} v
 */
function projectQ(right, left, v) {
    let sp;

    sp = Vector.scalar_product(left, v);
    v.lam_diff(sp, right);
}

/**
 * @param {Vector} right
 * @param {Vector} left
 * @param {Vector} v
 * @param {number} f
 */
function projectP(right, left, v, f = 1) {
    let i, n = v.size();
    let sp;

    sp = Vector.scalar_product(left, v) / f;
    v.test(right);
    for (i = 0; i < n; i++) v.c[i] = sp * right.c[i];
}

/**
 * @param {string} filename
 * @param {string} folder
 * @return string
 */
function getPath(filename, folder = "Data") {
    return path.join(__dirname, "..", folder, filename);
}

/**
 * @param {Vector} a
 * @param {Vector} b
 * @return number
 */
function diff_norm_rel(a, b) {
    let sum, ss;
    let i, n;
    n = a.size();
    sum = 0.0;
    //#pragma omp parallel for reduction(+:sum)
    for (i = 0; i < n; i++) {
        ss = Vector.abs(a.c[i]) + Vector.abs(b.c[i]);
        if (ss === 0) continue;
        sum += Vector.abs(a.c[i] - b.c[i]) / ss;
    }
    return sum;
}

/**
 * @param {Vector} a
 * @return number
 */
function pagerank_normalize(a) {
    let sum;

    sum = Vector.sum_vector(a);
    a.div_eq(sum);
    return sum;
}

/**
 * @param{Vector} pagerank
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{number} iprint
 * @param{Vector} node
 * @param{number} trans_frag
 * @return {Promise<number>}
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
    let a = new Vector(pagerank);
    quality_rel = 1e40;
    dlambda = 0;
    for (l = 0; l < node.dim; l++) {
        dlambda += pagerank.c[node.c[l]];
        pagerank.c[node.c[l]] = 0;
    }
    dlambda_old = dlambda;
    pnorm = pagerank_normalize(pagerank);
    if (trans_frag) dlambda = 1.0 - pnorm;
    for (i = 0; i <= max_iter; i++) {
        Vector.swap(a, pagerank);
        if (trans_frag) {
            net.GTmult(delta_alpha, pagerank, a);
        } else {
            net.GGmult(delta_alpha, pagerank, a);
        }
        //pnorm = pagerank_normalize(pagerank);
        //console.log(printf("--> %5d  %25.16f", i, pnorm));
        dlambda = 0;
        for (l = 0; l < node.dim; l++) {
            dlambda += pagerank.c[node.c[l]];
            pagerank.c[node.c[l]] = 0;
        }
        pnorm = pagerank_normalize(pagerank);
        if (trans_frag) dlambda = 1.0 - pnorm;

        if (i % iprint === 0 || i === max_iter) {
            quality = Vector.diff_norm1(pagerank, a);
            q1 = quality_rel;
            quality_rel = diff_norm_rel(pagerank, a);
            //      pnorm=pagerank_normalize(pagerank);
            //      pnorm=sum_vector(pagerank);
// #pragma omp critical(print)
            // {
            console.log(printf("%5d  %18.10lg  %18.10lg  %25.16lg  %18.10lg  %25.16lg",
                i, quality, quality_rel, dlambda, Math.abs(dlambda - dlambda_old), pnorm));
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
    {
        console.log(printf("Convergence at i = %d  with lambda = %25.16lg.\n", i, 1.0 - dlambda));
        //     fflush(stdout);
    }
    return dlambda;
}


/**
 * @param{Vector} pagerank
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{number} iprint
 * @param{Vector} node
 * @param{number} trans_flag
 * @return {Promise}
 */
function calc_pagerank_project_threaded(pagerank, net, delta_alpha, iprint, node, trans_flag = 0){
    return new Promise((resolve, reject) => {
        const worker = new Worker(
            getPath("Worker.js", "Source"),
            {
                workerData: {
                    data: {pagerank,net,delta_alpha,iprint,node,trans_flag},
                    task: 1
                }
            });
        worker.on("message", result=>{
            let {dlambda} = result;
            resolve(dlambda);
        });
        worker.on("error", reject);
        worker.on('exit', (code) => {
            if (code !== 0)
                reject(new Error(`stopped with  ${code} exit code`));
        });
    });
}

/**
 * @param{Vector} right
 * @param{Vector} left
 * @param{Vector} pg
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{Vector} node
 * @return Promise<number>
 */
async function compute_project(right, left, pg, net, delta_alpha, node) {
    const iprint = 10;
    console.log("compute_project()");
    let sp, dlambda1, dlambda2, dlambda3;
    let node0 = new Vector(0);

    right.put_value(1.0);
    left.put_value(1.0);
    pg.put_value(1.0);

    console.log("Starting timer for parallel calculations!");
    let l_timer = getTime();
// #pragma omp parallel sections
    {
// #pragma omp section
        let t2 = calc_pagerank_project_threaded(left, net, delta_alpha, iprint, node, 1);
// #pragma omp section
        let t1 = calc_pagerank_project_threaded(right, net, delta_alpha, iprint, node);
// #pragma omp section
        let t3 = calc_pagerank_project_threaded(pg, net, delta_alpha, iprint, node);

        dlambda1 = await t1;
        dlambda2 = await t2;
        dlambda3 = await t3;
    }
    console.log(`Parallel calculations took ${getTime() - l_timer} ms!`);

    sp = 1.0 / Vector.scalar_product(left, right);
    left.mul_eq(sp);
    sp = Vector.scalar_product(left, right);
// #pragma omp critical(print)
    {
        console.log(printf("dlambda = %24.16f   diff = %f\n",
            dlambda1, Math.abs(dlambda1 - dlambda2)));
        console.log(printf("TEST: psi_left^T * psi_right = %26.16f\n", sp));
        //     fflush(stdout);
    }

    return dlambda1;
}

function compute_GR_th(start, stop, max_iter, dlambda, delta_alpha, G_R, G_rr, G_pr, G_qr, G_I, psiL, psiR, net, node) {
    return new Promise((resolve, reject) => {
        const worker = new Worker(
            getPath("Worker.js", "Source"),
            {
                workerData: {
                    data: {start, stop,
                        max_iter, dlambda, delta_alpha,
                        G_R, G_rr, G_pr,
                        G_qr, G_I, psiL,
                        psiR, net, node},
                    task: 2
                }
            });
        worker.on("message", result=>{
            console.log(result);
            resolve(result);
        });
        worker.on("error", reject);
        worker.on('exit', (code) => {
            if (code !== 0)
                reject(new Error(`stopped with  ${code} exit code`));
        });
    });
}

/**
 * @param {number} total
 * @param {number} n
 * @return {number[]}
 */
function splits(total, n){
    const res = new Array(n);
    res.fill(Math.floor(total / n));
    const t = total - res.reduce((i, j)=> i + j, 0);
    for(let i = 0; i < t; i++) res[i] += 1;
    return res;
}

/**
 * @param{Matrix} G_R
 * @param{Matrix} G_rr
 * @param{Matrix} G_pr
 * @param{Matrix} G_qr
 * @param{Matrix} G_I
 * @param{Vector} psiL
 * @param{Vector} psiR
 * @param{Vector} pg
 * @param{Network} net
 * @param{number} delta_alpha
 * @param{Vector} node
 */
async function compute_GR(G_R, G_rr, G_pr,
                          G_qr, G_I, psiL,
                          psiR, pg, net,
                          delta_alpha, node) {
    let n = net.size;
    let nr = node.dim;
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

    console.log("Computation of left and right eigenvectors of G_ss");
    dlambda = await compute_project(psiR, psiL, pg, net, delta_alpha, node);

    let input = new Vector(n),
        output = new Vector(n),
        s = new Vector(n),
        t = new Vector(n),
        f = new Vector(n),
        f2 = new Vector(n);
    // note that the last line also fixes the default size of dvec to n
    // which is important in the private declaration below which implicitly
    // calls the default constructor of dvec for each thread

// #pragma omp parallel for schedule(dynamic) private(in, out, s, t, f, f2, j, l, quality)
    let nCpu = require("os").cpus().length;
    let schedule = splits(nr, nCpu);
    let threads = [];
    let acc = 0;
    for(let i = 0; i < nCpu; i++){
        threads.push(compute_GR_th(
            acc, acc + schedule[i],
            max_iter, dlambda, delta_alpha,
            G_R, G_rr, G_pr,
            G_qr, G_I, psiL,
            psiR, net, node))
        acc += schedule[i];
    }
    await Promise.all(threads);
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

    let node;
    const len = await fsp.readFile(getPath(nodefile))
        .then(data => {
            const lines = bsplit(data, Buffer.from("\n"));
            let len = parseInt(lines[0].toString());
            node = new Vector(len);
            let i = 0;
            for (const line of lines.slice(1)) {
                if (line.length === 0) continue;
                node.c[i++] = parseInt(line.toString());
            }
            return len;
        });
    console.log(printf("reading of nodefile finished: len = %d\n", len));
    const net = new Network(netfile);
    let GR = new Matrix(len, len),
        Grr = new Matrix(len, len),
        Gpr = new Matrix(len, len),
        Gqr = new Matrix(len, len),
        GI = new Matrix(len, len);
    const n = net.size;
    let psiL = new Vector(n),
        psiR = new Vector(n),
        pg = new Vector(n);
    nodefile = nodefile.split(".")[0];
    await compute_GR(GR, Grr, Gpr, Gqr, GI, psiL, psiR, pg, net, delta_alpha, node);
    Matrix.print_mat(Gqr, `Gqr_${net.base_name}_${nodefile}_${len}.dat`, nodefilenames);
    console.log(`Calculations took ${(getTime() - start) / 1000} sec\n`);
    return 0;
}

module.exports = main;
