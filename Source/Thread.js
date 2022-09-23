// noinspection DuplicatedCode

const Network = require("./Network");
const Matrix = require("./Matrix");
const printf = require("printf");
const Vector = require("./Vector");
const { parentPort } = require("worker_threads");

const eps_pagerank = 1e-13;

const {getTime, projectQ, projectP, diff_norm_rel, pagerank_normalize} = require("./Util");

/**
 * @param {Vector} pagerank
 * @param {Network} net
 * @param {number} delta_alpha
 * @param {number} iprint
 * @param {Vector} node
 * @param {number} trans_frag
 * @return {number}
 */
function calc_pagerank_project(pagerank, net, delta_alpha, iprint, node, trans_frag) {
    const func_t = getTime();
    let quality, quality_rel, q1, qfak, pnorm, dlambda, dlambda_old;
    let i, max_iter, l;

    if (iprint <= 0) iprint = 1;
    max_iter = Math.floor(-Math.log(eps_pagerank) / (delta_alpha + 3E-7));
    max_iter *= 2;

    // console.log(printf("max_iter = %d", max_iter));
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
    let iter_t = getTime();
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
            // pnorm=pagerank_normalize(pagerank);
            // pnorm=sum_vector(pagerank);
            // console.log(printf("%5d  %18.10lg  %18.10lg  %25.16lg  %18.10lg  %25.16lg",
            //     i, quality, quality_rel, dlambda, Math.abs(dlambda - dlambda_old), pnorm));
            console.log(`#${id} ${i}\t : ${getTime() - iter_t} ms`);
            parentPort.postMessage({
                done: false,
                data: `#${id} ${i}\t : ${getTime() - iter_t} ms`
            })
            iter_t = getTime();
            dlambda_old = dlambda;
            if (quality_rel < eps_pagerank) break;
            if (quality_rel < 1e-3) {
                if (quality_rel * qfak > q1) break;
            }
        }
    }
    // console.log(printf("Convergence at i = %d  with lambda = %25.16lg.\n", i, 1.0 - dlambda));
    console.log(`#${id} calc_pg_proj : ${getTime() - func_t} ms`);
    parentPort.postMessage({
        done: false,
        data: `#${id} calc_pg_proj : ${getTime() - func_t} ms`
    })
    return dlambda;
}

let input, output, s, t, f, f2;

let G_R, G_rr, G_pr, G_qr, G_I, psiL, psiR, pg, net, delta_alpha, node, max_iter, dlambda;
let nr, n;
function compute_GR_heavy(i){
    let iter_timer = getTime();
    let quality;
    let j, l;
    input.put_value(0.0);
    input.c[node.c[i]] = 1;
    net.GGmult(delta_alpha, output, input);
    input.c[node.c[i]] = 0;
    for (j = 0; j < nr; j++) {
        G_R.mat[j][i] = output.c[node.c[j]];
        G_rr.mat[j][i] = output.c[node.c[j]];
        output.c[node.c[j]] = 0;
    }
    // s = output;
    s.eq(output);
    projectP(psiR, psiL, output, dlambda);
    projectQ(psiR, psiL, s);
    // f = s;
    f.eq(s);
    let inner_t = getTime();
    for (l = 0; l < max_iter; l++) {
        t.eq(s);
        net.GGmult(delta_alpha, f2, f, 0);
        Vector.swap(f, f2);
        for (j = 0; j < nr; j++) f.c[node.c[j]] = 0;
        projectQ(psiR, psiL, f);
        // s += f;
        s.add_eq(f);
        quality = Vector.diff_norm1(t, s);
        if (l % 10 === 0) {
            // console.log(printf("%5d  %5d  %18.10lg  %18.10lg", i, l, quality, Vector.norm1(f)));
            // console.log(`#${id} compute_GR sub-iter i=${i}\tl=${l}\t : ${getTime() - inner_t} ms`);
            parentPort.postMessage({
                done: false,
                data: `#${id} compute_GR sub-iter i=${i}\tl=${l}\t : ${getTime() - inner_t} ms`
            });
            inner_t = getTime();
        }
        if (quality <= 0) break;
    }
    net.GGmult(delta_alpha, f, output, 0);
    for (j = 0; j < nr; j++) {
        G_pr.mat[j][i] = f.c[node.c[j]];
    }
    net.GGmult(delta_alpha, f, s, 0);
    for (j = 0; j < nr; j++) {
        G_qr.mat[j][i] = f.c[node.c[j]];
    }
    output.add_eq(s);
    net.GGmult(delta_alpha, f, output, 0);
    for (j = 0; j < nr; j++) {
        G_I.mat[j][i] = f.c[node.c[j]];
        G_R.mat[j][i] += f.c[node.c[j]];
    }
    // console.log(printf("%5d  Convergence: %5d  %5d  %18.10lg  %18.10lg\n",
    //     i, i, l, quality, Vector.norm1(f)));
    // console.log(`#${id} compute_GR iter i=${i}\t : ${getTime() - iter_timer} ms`);
    parentPort.postMessage({
        done: false,
        data: `#${id} compute_GR iter i=${i}\t : ${getTime() - iter_timer} ms`
    });
}

let id;
function processor(msg){
    if(msg.options.work){
        switch (msg.options.task) {
            case 1: {
                id = msg.options.id;
                let {pagerank, net, delta_alpha, iprint, node, trans_frag} = msg.data;
                pagerank = Vector.fromObj(pagerank);
                net = Network.fromObj(net);
                node = Vector.fromObj(node);
                let dlambda = calc_pagerank_project(pagerank, net, delta_alpha, iprint, node, trans_frag);
                parentPort.postMessage({
                    done: true,
                    data: dlambda
                });
                break;
            }
            case 2: {
                if(msg.options.stage === 1){
                    let timer = getTime();
                    id = msg.options.id;
                    G_R = Matrix.fromObj(msg.data.G_R);
                    G_rr = Matrix.fromObj(msg.data.G_rr);
                    G_pr = Matrix.fromObj(msg.data.G_pr);
                    G_qr = Matrix.fromObj(msg.data.G_qr);
                    G_I = Matrix.fromObj(msg.data.G_I);
                    psiL = Vector.fromObj(msg.data.psiL);
                    psiR = Vector.fromObj(msg.data.psiR);
                    pg = Vector.fromObj(msg.data.pg);
                    net = Network.fromObj(msg.data.net);
                    node = Vector.fromObj(msg.data.node);
                    delta_alpha = msg.data.delta_alpha;
                    max_iter = msg.data.max_iter;
                    dlambda = msg.data.dlambda;
                    n = net.size;
                    nr = node.dim;
                    input = new Vector(n);
                    output = new Vector(n);
                    s = new Vector(n);
                    t = new Vector(n);
                    f = new Vector(n);
                    f2 = new Vector(n);
                    let delay = getTime() - timer;
                    parentPort.postMessage({id, delay});
                }
                if(msg.options.stage === 2){

                    let {i} = msg.data;
                    compute_GR_heavy(i);

                    parentPort.postMessage({done: true, id});
                }
                break;
            }
        }
        if(msg.options.once) parentPort.off("message", processor);
    } else parentPort.off("message", processor);
}

parentPort.on("message", processor);

