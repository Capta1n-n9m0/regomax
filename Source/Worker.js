// noinspection DuplicatedCode

const fsp = require("fs").promises;
const path = require("path");
const bsplit = require("buffer-split");
const Network = require("./Network");
const Matrix = require("./Matrix");
const printf = require("printf");
const Vector = require("./Vector");
const { workerData, parentPort } = require('worker_threads');


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
function processFilename(filename, folder = "Data") {
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
 * @param{number} trans_flag
 * @return {number}
 */
function calc_pagerank_project(pagerank, net, delta_alpha, iprint, node, trans_flag) {
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
    if (trans_flag) dlambda = 1.0 - pnorm;
    for (i = 0; i <= max_iter; i++) {
        Vector.swap(a, pagerank);
        if (trans_flag) {
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
        if (trans_flag) dlambda = 1.0 - pnorm;

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

if(workerData.task === 1){
    // calc_pagerank_project
    let {pagerank, net, delta_alpha, iprint, node, trans_flag} = workerData.data;
    let pg = Vector.fromObj(pagerank);
    net = Network.fromObj(net);
    const dlambda = calc_pagerank_project(pg, net, delta_alpha, iprint, node, trans_flag);
    parentPort.postMessage({dlambda});
} else if(workerData.task === 2){


    let {start, stop, input, output, s, t, f, f2, max_iter, dlambda, delta_alpha, G_R, G_rr, G_pr, G_qr, G_I, psiL, psiR, net, node} = workerData.data;

    let nr = stop
    input = Vector.fromObj(input);
    output = Vector.fromObj(output);
    s = Vector.fromObj(s);
    t = Vector.fromObj(t);
    f = Vector.fromObj(f);
    f2 = Vector.fromObj(f2);
    G_R = Matrix.fromObj(G_R);
    G_rr = Matrix.fromObj(G_rr);
    G_pr = Matrix.fromObj(G_pr);
    G_qr = Matrix.fromObj(G_qr);
    G_I = Matrix.fromObj(G_I);
    psiL = Vector.fromObj(psiL);
    psiR = Vector.fromObj(psiR);
    net = Network.fromObj(net);
    node = Vector.fromObj(node);

    let i, j, l;
    let quality;
    for (i = start; i < nr; i++) {
        input.put_value(0.0);
        input.c[node.c[i]] = 1;
        net.GGmult(delta_alpha, output, input);
        input.c[node.c[i]] = 0;
        for (j = 0; j < nr; j++) {
            G_R.mat[j].c[i] = output.c[node.c[j]];
            G_rr.mat[j].c[i] = output.c[node.c[j]];
            output.c[node.c[j]] = 0;
        }
        // s = output;
        s.eq(output);
        projectP(psiR, psiL, output, dlambda);
        projectQ(psiR, psiL, s);
        // f = s;
        f.eq(s);

        for (l = 0; l < max_iter; l++) {
            t.eq(s);
            net.GGmult(delta_alpha, f2, f, 0);
            Vector.swap(f, f2);
            for (j = 0; j < nr; j++) f.c[node.c[j]] = 0;
            projectQ(psiR, psiL, f);
            // s += f;
            s.add_eq(f);
            quality = Vector.diff_norm1(t, s);
// #pragma omp critical(print)
            {
                if (l % 10 === 0) {
                    console.log(printf("%5d  %5d  %18.10lg  %18.10lg", i, l, quality, Vector.norm1(f)));
                    //         fflush(stdout);
                }
            }
            if (quality <= 0) break;
        }
// #pragma omp critical(print)
        {
            console.log(printf("%5d  Convergence: %5d  %5d  %18.10lg  %18.10lg\n",
                i, i, l, quality, Vector.norm1(f)));
            //     fflush(stdout);
        }
        net.GGmult(delta_alpha, f, output, 0);
        for (j = 0; j < nr; j++) {
            G_pr.mat[j].c[i] = f.c[node.c[j]];
        }
        net.GGmult(delta_alpha, f, s, 0);
        for (j = 0; j < nr; j++) {
            G_qr.mat[j].c[i] = f.c[node.c[j]];
        }
        output.add_eq(s);
        net.GGmult(delta_alpha, f, output, 0);
        for (j = 0; j < nr; j++) {
            G_I.mat[j].c[i] = f.c[node.c[j]];
            G_R.mat[j].c[i] += f.c[node.c[j]];
        }
    }
    parentPort.postMessage({data: "Done!"});
}


