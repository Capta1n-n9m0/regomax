/**
 * @return {number}
 */
function getTime() {
    const delay = Number(process.hrtime.bigint() / BigInt(1_000));
    return delay / 1_000;
}

module.exports = {getTime};