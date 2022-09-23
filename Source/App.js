// noinspection DuplicatedCode

const fsp = require("fs").promises;
const path = require("path");
const bsplit = require("buffer-split");
const Network = require("./Network");
const Matrix = require("./Matrix");
const printf = require("printf");
const Vector = require("./Vector");

const eps_pagerank = 1e-13;

const {getTime, projectQ, projectP, diff_norm_rel, pagerank_normalize} = require("./Util");



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
            console.log(`#${0} ${i}\t : ${getTime() - iter_t} ms`);
            iter_t = getTime();
            dlambda_old = dlambda;
            if (quality_rel < eps_pagerank) break;
            if (quality_rel < 1e-3) {
                if (quality_rel * qfak > q1) break;
            }
        }
    }
    // console.log(printf("Convergence at i = %d  with lambda = %25.16lg.\n", i, 1.0 - dlambda));
    console.log(`#${0} calc_pg_proj : ${getTime() - func_t} ms`);
    return dlambda;
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
    let f_timer = getTime();
    const iprint = 10;
    let sp, dlambda1, dlambda2, dlambda3;
    let node0 = new Vector(0);

    right.put_value(1.0);
    left.put_value(1.0);
    pg.put_value(1.0);

    let p1 = calc_pagerank_project(right, net, delta_alpha, iprint, node);
    let p2 = calc_pagerank_project(left, net, delta_alpha, iprint, node, 1);
    let p3 = calc_pagerank_project(pg, net, delta_alpha, iprint, node0);

    dlambda1 = await p1;
    dlambda2 = await p2;
    dlambda3 = await p3;

    sp = 1.0 / Vector.scalar_product(left, right);
    left.mul_eq(sp);

    //sp = Vector.scalar_product(left, right);
    // console.log(printf("dlambda = %24.16f   diff = %f\n",
    //     dlambda1, Math.abs(dlambda1 - dlambda2)));
    // console.log(printf("TEST: psi_left^T * psi_right = %26.16f\n", sp));

    console.log(`compute_project : ${getTime() - f_timer} ms`);
    return dlambda1;
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

    let c_GR_h_timer = getTime();
    let iter_timer = getTime();
    for (i = 0; i < nr; i++) {
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
                //console.log(printf("%5d  %5d  %18.10lg  %18.10lg", i, l, quality, Vector.norm1(f)));
                console.log(`#${0} compute_GR sub-iter i=${i}\tl=${l}\t : ${getTime() - inner_t} ms`);
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
        console.log(`#${0} compute_GR iter i=${i}\t : ${getTime() - iter_timer} ms`);
        iter_timer = getTime();
    }
    console.log(`compute_GR loop : ${getTime() - c_GR_h_timer} ms`);
}

async function main(argv) {
    const start_timer = getTime();
    const net_file = argv[2];
    const delta_alpha = parseFloat(argv[3]);
    const iprint = parseInt(argv[4]);
    const print_number = parseInt(argv[5]);
    const ten_number = parseInt(argv[6]);
    let node_file = argv[7];
    const nodenames_file = argv[8];
    const limit = parseInt(argv[9]);
    console.log(`input-file     = ${path.resolve(net_file)}`);
    console.log(`1-alpha        = ${delta_alpha}`);
    console.log(`iprint         = ${iprint}`);
    console.log(`print_number   = ${print_number}`);
    console.log(`ten_number     = ${ten_number}`);
    console.log(`node_file      = ${path.resolve(node_file)}`);
    console.log(`nodenames_file = ${path.resolve(nodenames_file)}`);
    
    
    let nodes_read_timer = getTime();
    let tmp = [];
    const len = await fsp.readFile(node_file)
        .then(data => {
            const lines = bsplit(data, Buffer.from("\n"));
            for (const line of lines.slice(1)) {
                if (line.length === 0) continue;
                const n = parseInt(line.toString());
                if(limit) {
                    if (n <= limit) tmp.push(n);
                }
                else tmp.push(n);
            }
            return tmp.length;
        });
    let node = new Vector(len);
    for(let i = 0; i < tmp.length; i++){
        node.c[i] = tmp[i];
    }
    console.log(`Read node_file : ${len} nodes`);
    console.log(`Read node_file : ${getTime() - nodes_read_timer} ms`);
    const net = new Network(net_file, limit);
    let GR = new Matrix(len, len),
        Grr = new Matrix(len, len),
        Gpr = new Matrix(len, len),
        Gqr = new Matrix(len, len),
        GI = new Matrix(len, len);
    const n = net.size;
    let psiL = new Vector(n),
        psiR = new Vector(n),
        pg = new Vector(n);
    node_file = path.basename(node_file, ".nodes");
    let calc_timer = getTime();
    await compute_GR(GR, Grr, Gpr, Gqr, GI, psiL, psiR, pg, net, delta_alpha, node);
    console.log(`Calculations : ${getTime() - calc_timer} ms`);
    let write_timer = getTime();
    Matrix.print_mat(Gqr, `Gqr_${net.base_name}_${node_file}_${len}.dat`, nodenames_file);
    console.log(`Writing matrix : ${getTime() - write_timer} ms`);
    console.log(`Execution : ${getTime() - start_timer} ms\n`);
    return 0;
}

module.exports = main;
